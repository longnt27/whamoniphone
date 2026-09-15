#!/usr/bin/env python3
"""Export the recurrent WHAM subset with the image-feature path connected.

This intentionally exports only the modules that can be evaluated without the
licensed SMPL body-model assets.  It uses the official WHAM checkpoint for the
motion encoder, trajectory decoder, feature integrator, and motion decoder.

Example:
    python utils/export_wham_image_step.py \
      --wham-repo /path/to/yohanshin/WHAM \
      --checkpoint checkpoints/wham_vit_bedlam_w_3dpw.pth.tar \
      --output WhamApp/WhamApp/WHAM_ImageStep.mlpackage
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn


class WHAMImageStep(nn.Module):
    """One causal WHAM frame, with image integration and explicit state I/O."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(
        self,
        x_step,
        keypoint_mask_step,
        image_feature_step,
        image_feature_valid_step,
        cam_a_step,
        prev_kp3d,
        prev_root,
        prev_pose,
        h_enc,
        c_enc,
        h_traj,
        c_traj,
        h_dec,
        c_dec,
    ):
        # Match Network.preprocess without boolean advanced indexing, which is
        # difficult for Core ML to lower. A mask value of one means "missing".
        mask = keypoint_mask_step.clamp(0.0, 1.0)
        coordinate_mask = torch.repeat_interleave(mask, 2, dim=-1)
        full_mask = torch.cat((coordinate_mask, torch.zeros_like(x_step[..., :3])), dim=-1)
        learned_mask = (mask.unsqueeze(-1) * self.network.mask_embedding).reshape(1, 1, 34)
        learned_mask = torch.cat((learned_mask, torch.zeros_like(x_step[..., :3])), dim=-1)
        x_step = x_step * (1.0 - full_mask) + learned_mask

        x_emb = self.network.motion_encoder.embed_layer(x_step.reshape(1, 1, -1))
        (pred_kp3d,), encoded_context, (h_enc_out, c_enc_out) = (
            self.network.motion_encoder.regressor(x_emb, [prev_kp3d], (h_enc, c_enc))
        )
        motion_context = torch.cat((encoded_context, pred_kp3d), dim=-1)

        (pred_vel, pred_root), _, (h_traj_out, c_traj_out) = (
            self.network.trajectory_decoder.regressor(
                motion_context, [prev_root, cam_a_step], (h_traj, c_traj)
            )
        )

        integrator_input = torch.cat((motion_context, image_feature_step), dim=-1)
        integrated = self.network.integrator.layer1(integrator_input)
        integrated = self.network.integrator.relu1(integrated)
        integrated = self.network.integrator.layer2(integrated)
        integrated = self.network.integrator.relu2(integrated)
        integrated = self.network.integrator.layer3(integrated)

        # With a valid image this is the official residual integration. When a
        # detector misses, bypass the image MLP explicitly instead of silently
        # feeding a zero feature through it.
        valid = image_feature_valid_step.clamp(0.0, 1.0)
        fused_context = valid * (integrated + motion_context) + (1.0 - valid) * motion_context

        (pred_pose, pred_shape, pred_cam, pred_contact), _, (h_dec_out, c_dec_out) = (
            self.network.motion_decoder.regressor(
                fused_context, [prev_pose], (h_dec, c_dec)
            )
        )

        return (
            pred_kp3d,
            pred_root,
            pred_vel,
            pred_pose,
            pred_shape,
            pred_cam,
            pred_contact,
            h_enc_out,
            c_enc_out,
            h_traj_out,
            c_traj_out,
            h_dec_out,
            c_dec_out,
        )


def load_network(wham_repo: Path, checkpoint_path: Path) -> nn.Module:
    sys.path.insert(0, str(wham_repo))
    from lib.models.wham import Network  # pylint: disable=import-outside-toplevel

    network = Network(
        smpl=None,
        pose_dr=0.1,
        d_embed=512,
        n_layers=3,
        d_feat=1024,
        rnn_type="LSTM",
    ).cpu().eval()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_state = checkpoint["model"]
    usable_state = {key: value for key, value in model_state.items() if not key.startswith("smpl.")}
    missing, unexpected = network.load_state_dict(usable_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    for module in network.modules():
        if isinstance(module, torch.nn.LSTM):
            module.dropout = 0.0
            module.flatten_parameters()
    return network


def export_model(network: nn.Module, output_path: Path) -> None:
    wrapper = WHAMImageStep(network).eval()
    h_enc = torch.zeros(3, 1, 512)
    h_other = torch.zeros(3, 1, 563)
    example = (
        torch.randn(1, 1, 37),
        torch.zeros(1, 1, 17),
        torch.randn(1, 1, 1024),
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 6),
        torch.randn(1, 1, 51),
        torch.randn(1, 1, 6),
        torch.randn(1, 1, 144),
        h_enc,
        h_enc.clone(),
        h_other,
        h_other.clone(),
        h_other.clone(),
        h_other.clone(),
    )
    traced = torch.jit.trace(wrapper, example, check_trace=True)

    names = (
        "x_step",
        "keypoint_mask_step",
        "image_feature_step",
        "image_feature_valid_step",
        "cam_a_step",
        "prev_kp3d",
        "prev_root",
        "prev_pose",
        "h_enc_in",
        "c_enc_in",
        "h_traj_in",
        "c_traj_in",
        "h_dec_in",
        "c_dec_in",
    )
    output_names = (
        "pred_kp3d",
        "pred_root",
        "pred_vel",
        "pred_pose",
        "pred_shape",
        "pred_cam",
        "pred_contact",
        "h_enc_out",
        "c_enc_out",
        "h_traj_out",
        "c_traj_out",
        "h_dec_out",
        "c_dec_out",
    )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=value.shape) for name, value in zip(names, example)],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.author = "WHAM-iOS; weights from the official WHAM checkpoint"
    mlmodel.license = "WHAM research license; see https://github.com/yohanshin/WHAM"
    mlmodel.short_description = "Causal WHAM step with 1024-D image feature integration"

    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Conversion smoke test and a hard assertion that image features affect the
    # motion-decoder output. This catches the exact regression this project had.
    compiled = ct.models.MLModel(str(output_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    base = {name: value.detach().numpy().astype(np.float32) for name, value in zip(names, example)}
    out_a = compiled.predict(base)
    changed = dict(base)
    changed["image_feature_step"] = -base["image_feature_step"]
    out_b = compiled.predict(changed)
    delta = float(np.max(np.abs(out_a["pred_pose"] - out_b["pred_pose"])))
    if not np.isfinite(delta) or delta < 1e-5:
        raise RuntimeError(f"Image feature influence check failed (max pose delta={delta})")
    print(f"Saved {output_path}")
    print(f"Image feature influence: max |pose delta| = {delta:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.wham_repo / "lib/models/wham.py").is_file():
        parser.error("--wham-repo is not an official WHAM checkout")
    if not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")

    network = load_network(args.wham_repo.resolve(), args.checkpoint.resolve())
    export_model(network, args.output.resolve())


if __name__ == "__main__":
    main()
