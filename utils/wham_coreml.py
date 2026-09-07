"""Streaming WHAM export with image-feature integration; no SMPL forward pass.

Run --help for the CPU-only export command. Importing this module only needs
PyTorch; upstream WHAM and coremltools are imported by the export command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

FEATURE_DIM = 1024
INIT_OUTPUTS = ("h_enc", "c_enc", "h_traj", "c_traj", "h_dec", "c_dec")
STEP_INPUTS = (
    "x_step", "features_step", "cam_a_step", "prev_kp3d", "prev_root", "prev_pose",
    "h_enc_in", "c_enc_in", "h_traj_in", "c_traj_in", "h_dec_in", "c_dec_in",
)
STEP_OUTPUTS = (
    "pred_kp3d", "pred_root", "pred_vel", "pred_pose", "pred_shape", "pred_cam",
    "pred_contact", "h_enc_out", "c_enc_out", "h_traj_out", "c_traj_out",
    "h_dec_out", "c_dec_out",
)


class WHAMInit(nn.Module):
    def __init__(self, network: nn.Module, main_joints: list[int]):
        super().__init__()
        self.encoder_init = network.motion_encoder.neural_init
        self.decoder_init = network.motion_decoder.neural_init
        self.main_joints = list(main_joints)
        rnn = network.trajectory_decoder.regressor.rnn
        if not isinstance(rnn, nn.LSTM):
            raise ValueError("The explicit h/c export requires an LSTM checkpoint.")
        self.trajectory_layers = rnn.num_layers
        self.trajectory_hidden = rnn.hidden_size

    def forward(self, init_kp, init_smpl):
        batch = init_kp.shape[0]
        h_enc, c_enc = self.encoder_init(init_kp.reshape(batch, 1, -1))
        shape = (self.trajectory_layers, batch, self.trajectory_hidden)
        h_traj = init_kp.new_zeros(shape)
        c_traj = init_kp.new_zeros(shape)
        pose = init_smpl.reshape(batch, 1, 24, 6)
        h_dec, c_dec = self.decoder_init(
            pose[:, :, self.main_joints].reshape(batch, 1, -1)
        )
        return h_enc, c_enc, h_traj, c_traj, h_dec, c_dec


class WHAMStep(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        if network.integrator is None:
            raise ValueError("The checkpoint must include the trained image integrator.")
        self.encoder = network.motion_encoder
        self.trajectory = network.trajectory_decoder
        self.integrator = network.integrator
        self.decoder = network.motion_decoder
        context_dim = self.encoder.regressor.rnn.hidden_size + 51
        image_dim = self.integrator.layer1.in_features - context_dim
        if image_dim != FEATURE_DIM:
            raise ValueError(f"Expected ViT features of width 1024, got {image_dim}.")

    def integrate(self, context, features):
        # Preserve upstream Integrator weights and its exact residual condition.
        # Boolean indexed assignment in upstream is replaced by a static-shape
        # where operation to avoid dynamic gather/scatter in the Core ML graph.
        module = self.integrator
        fused = module.dr1(module.relu1(module.layer1(torch.cat((context, features), -1))))
        fused = module.dr2(module.relu2(module.layer2(fused)))
        fused = module.layer3(fused)
        mask = (features != 0).all(dim=-1).all(dim=-1).reshape(-1, 1, 1)
        return torch.where(mask, fused + context, fused)

    def forward(self, x_step, features_step, cam_a_step, prev_kp3d, prev_root,
                prev_pose, h_enc, c_enc, h_traj, c_traj, h_dec, c_dec):
        embedded = self.encoder.pos_drop(self.encoder.embed_layer(x_step))
        (kp3d,), context, (he, ce) = self.encoder.regressor(
            embedded, [prev_kp3d], (h_enc, c_enc)
        )
        context = torch.cat((context, kp3d), dim=-1)
        # As in upstream WHAM, the trajectory branch stays image-independent.
        (velocity, root), _, (ht, ct) = self.trajectory.regressor(
            context, [prev_root, cam_a_step], (h_traj, c_traj)
        )
        fused = self.integrate(context, features_step)
        (pose, shape, camera, contact), _, (hd, cd) = self.decoder.regressor(
            fused, [prev_pose], (h_dec, c_dec)
        )
        return kp3d, root, velocity, pose, shape, camera, contact, he, ce, ht, ct, hd, cd


def load_weights(network: nn.Module, checkpoint: dict) -> None:
    """Fail on missing trained weights, especially integrator.*; exclude only SMPL."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected an upstream WHAM checkpoint containing a 'model' state dict.")
    state = {key: value for key, value in checkpoint["model"].items()
             if not key.startswith("smpl.")}
    network.load_state_dict(state, strict=True)


def reference_step(network, args):
    """One-frame reference using the unmodified upstream Integrator.forward."""
    x, features, cam, kp, root, pose, he, ce, ht, ct, hd, cd = args
    encoder = network.motion_encoder
    (kp,), context, (he, ce) = encoder.regressor(
        encoder.pos_drop(encoder.embed_layer(x)), [kp], (he, ce)
    )
    context = torch.cat((context, kp), -1)
    (velocity, root), _, (ht, ct) = network.trajectory_decoder.regressor(
        context, [root, cam], (ht, ct)
    )
    fused = network.integrator(context, features)
    (pose, shape, camera, contact), _, (hd, cd) = network.motion_decoder.regressor(
        fused, [pose], (hd, cd)
    )
    return kp, root, velocity, pose, shape, camera, contact, he, ce, ht, ct, hd, cd


def feedback(args, outputs, x, features):
    """Carry every backend's own predictions and six state tensors forward."""
    return (x, features, args[2], outputs[0], outputs[1], outputs[3], *outputs[7:])


def check_wiring(network, step, args):
    """Checkpoint-level diagnostic; this does not measure pose accuracy."""
    with torch.no_grad():
        for features in (torch.ones_like(args[1]), torch.zeros_like(args[1])):
            current = (args[0], features, *args[2:])
            for actual, expected in zip(step(*current), reference_step(network, current)):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        first = step(*args)
        alternate = step(args[0], -args[1], *args[2:])
        for index in (0, 1, 2, 7, 8, 9, 10):
            torch.testing.assert_close(first[index], alternate[index], rtol=0, atol=0)
        deltas = {STEP_OUTPUTS[i]: (first[i] - alternate[i]).abs().max().item()
                  for i in (3, 4, 5, 6)}
        if deltas["pred_pose"] == 0 or deltas["pred_shape"] == 0:
            raise RuntimeError(f"Image sensitivity check failed: {deltas}")
        return deltas


def export(network, main_joints, output_dir: Path, verify_coreml: bool = False):
    import coremltools as ct
    import numpy as np

    network = network.cpu().eval()
    torch.manual_seed(27)
    init = WHAMInit(network, main_joints).eval()
    step = WHAMStep(network).eval()
    initial = (torch.randn(1, 1, 88), torch.randn(1, 1, 144))
    with torch.no_grad():
        states = init(*initial)
        args = (torch.randn(1, 1, 37), torch.ones(1, 1, FEATURE_DIM),
                torch.zeros(1, 1, 6), initial[0][..., :51], torch.zeros(1, 1, 6),
                initial[1], *states)
        deltas = check_wiring(network, step, args)
        traced_init = torch.jit.trace(init, initial)
        # Check zero features too: the residual mask must not freeze during tracing.
        zero_args = (args[0], torch.zeros_like(args[1]), *args[2:])
        traced_step = torch.jit.trace(step, args, check_inputs=[zero_args])

    options = dict(convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                   compute_precision=ct.precision.FLOAT16)
    ml_init = ct.convert(traced_init, inputs=[
        ct.TensorType(name=name, shape=value.shape, dtype=np.float32)
        for name, value in zip(("init_kp", "init_smpl"), initial)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in INIT_OUTPUTS], **options)
    ml_step = ct.convert(traced_step, inputs=[
        ct.TensorType(name=name, shape=value.shape, dtype=np.float32)
        for name, value in zip(STEP_INPUTS, args)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in STEP_OUTPUTS], **options)
    ml_step.user_defined_metadata["wham.image_features"] = "features_step:1x1x1024;v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    ml_init.save(str(output_dir / "WHAM_I.mlpackage"))
    ml_step.save(str(output_dir / "WHAM_S.mlpackage"))
    report = {"image_sensitivity_max_abs": deltas, "coreml_sequence_verified": False}

    if verify_coreml:
        # Reload CPU-only to make the first conversion check reproducible.
        ml_init = ct.models.MLModel(str(output_dir / "WHAM_I.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
        ml_step = ct.models.MLModel(str(output_dir / "WHAM_S.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
        native_states = ml_init.predict({name: value.numpy() for name, value in
                                        zip(("init_kp", "init_smpl"), initial)})
        native_args = (*args[:6], *(torch.from_numpy(native_states[name]) for name in INIT_OUTPUTS))
        reference_args = args
        maxima = {name: 0.0 for name in STEP_OUTPUTS}
        with torch.no_grad():
            for frame in range(16):
                x = torch.randn_like(args[0])
                features = (torch.zeros_like(args[1]) if frame % 4 == 0 else torch.randn_like(args[1]))
                reference_args = (x, features, *reference_args[2:])
                native_args = (x, features, *native_args[2:])
                expected = step(*reference_args)
                prediction = ml_step.predict({name: value.numpy() for name, value in
                                              zip(STEP_INPUTS, native_args)})
                actual = tuple(torch.from_numpy(prediction[name]) for name in STEP_OUTPUTS)
                for name, got, wanted in zip(STEP_OUTPUTS, actual, expected):
                    maxima[name] = max(maxima[name], (got - wanted).abs().max().item())
                    # Numerical smoke gate, not a validated task-accuracy tolerance.
                    torch.testing.assert_close(got, wanted, rtol=2e-2, atol=2e-2)
                reference_args = feedback(reference_args, expected, x, features)
                native_args = feedback(native_args, actual, x, features)
        report.update(coreml_sequence_verified=True, coreml_max_abs_error=maxima)
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wham-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/yamls/model_base.yaml")
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow pickle deserialization only for a checkpoint you trust.")
    parser.add_argument("--verify-coreml", action="store_true",
                        help="On macOS, also compare a 16-step Core ML/PyTorch recurrence.")
    args = parser.parse_args()
    root = args.wham_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (root / "lib/models/wham.py").is_file():
        parser.error("--wham-root must point to the upstream WHAM checkout.")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.verify_coreml and sys.platform != "darwin":
        parser.error("--verify-coreml requires native macOS Core ML.")
    sys.path.insert(0, str(root))
    import yaml
    from configs import constants
    from lib.models.wham import Network

    with (root / args.model_config).open() as handle:
        config = yaml.safe_load(handle)
    config["d_feat"] = FEATURE_DIM
    # Only the learned layers are exported. No SMPL data or CUDA setup is needed.
    network = Network(smpl=None, **config).cpu().eval()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=not args.trust_checkpoint)
    load_weights(network, weights)
    export(network, constants.BMODEL.MAIN_JOINTS, args.out, args.verify_coreml)


if __name__ == "__main__":
    main()
