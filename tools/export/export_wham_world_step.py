#!/usr/bin/env python3
"""Export the frozen SMPL + causal WHAM trajectory-refiner tail.

The existing WHAM_ImageStep package ends at pose, shape, contact, root and
velocity.  This package consumes those outputs, performs neutral SMPL linear
blend skinning, applies WHAM's contact velocity correction and two-layer
trajectory refiner one frame at a time, rolls out world translation, and emits
the current world-space mesh plus all explicit recurrent state.

No training occurs.  The weights come from the same released WHAM checkpoint;
the neutral SMPL buffers come from the released HMR2.0-S checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from export_wham_image_step import load_network
from hmr2s_frozen import HMR2S_CHECKPOINT_SHA256, checkpoint_smpl_buffers
from smplx.lbs import lbs
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    rows = rotation.reshape(-1, 2, 3)
    first = torch.nn.functional.normalize(rows[:, 0], dim=-1)
    second_raw = rows[:, 1]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


class WHAMWorldStep(nn.Module):
    """One causal SMPL/refiner/world rollout step with explicit state I/O."""

    def __init__(
        self,
        trajectory_refiner: nn.Module,
        smpl_buffers: dict[str, torch.Tensor],
        wham_joint_regressor: torch.Tensor,
        feet_regressor: torch.Tensor,
    ) -> None:
        super().__init__()
        self.refiner = trajectory_refiner.refiner
        for name, value in smpl_buffers.items():
            self.register_buffer(name, value)
        self.register_buffer("wham_joint_regressor", wham_joint_regressor.float())
        self.register_buffer("feet_regressor", feet_regressor.float())

    def forward(
        self,
        pred_pose: torch.Tensor,
        pred_shape: torch.Tensor,
        pred_contact: torch.Tensor,
        pred_root: torch.Tensor,
        pred_vel: torch.Tensor,
        pred_kp3d: torch.Tensor,
        h_enc_out: torch.Tensor,
        prev_unrefined_root: torch.Tensor,
        prev_unrefined_translation: torch.Tensor,
        prev_body_feet: torch.Tensor,
        prev_world_feet: torch.Tensor,
        prev_refined_root: torch.Tensor,
        prev_refined_translation: torch.Tensor,
        h_refiner_in: torch.Tensor,
        c_refiner_in: torch.Tensor,
        has_previous: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pose_matrices = rotation_6d_to_matrix(pred_pose).reshape(1, 24, 3, 3)
        vertices, _ = lbs(
            pred_shape.reshape(1, 10),
            pose_matrices,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )
        joints = torch.einsum("jv,bvc->bjc", self.wham_joint_regressor, vertices)
        feet = torch.einsum("jv,bvc->bjc", self.feet_regressor, vertices)
        pelvis = joints[:, [11, 12]].mean(dim=1, keepdim=True)
        vertices = vertices - pelvis
        joints = joints - pelvis
        feet = feet - pelvis

        previous_root_matrix = rotation_6d_to_matrix(prev_unrefined_root).reshape(
            1, 3, 3
        )
        current_root_matrix = rotation_6d_to_matrix(pred_root).reshape(1, 3, 3)
        camera_root_matrix = pose_matrices[:, 0]
        # WHAM's ordinary rollout rotates this frame's root-space velocity by
        # the previous root orientation (rollout_global_motion).
        rollout_velocity_world = torch.matmul(
            previous_root_matrix, pred_vel.reshape(1, 3, 1)
        ).squeeze(-1)
        unrefined_translation = (
            prev_unrefined_translation.reshape(1, 3) + rollout_velocity_world
        )
        camera_to_world_body = torch.matmul(
            current_root_matrix, camera_root_matrix.transpose(-1, -2)
        )
        body_feet = torch.matmul(
            camera_to_world_body.unsqueeze(1), feet.unsqueeze(-1)
        ).squeeze(-1)
        world_feet = body_feet + unrefined_translation.unsqueeze(1)

        previous = has_previous.reshape(1, 1, 1).clamp(0.0, 1.0)
        # reset_root_velocity intentionally uses the current root orientation,
        # which is distinct from the rollout rule above.
        reset_velocity_world = torch.matmul(
            current_root_matrix, pred_vel.reshape(1, 3, 1)
        ).squeeze(-1)
        foot_delta = (
            body_feet
            - prev_body_feet.reshape(1, 4, 3)
            + reset_velocity_world.unsqueeze(1)
        ) * previous
        stationary = (pred_contact.reshape(1, 4) > 0.5).to(foot_delta).unsqueeze(-1)
        stationary_velocity = foot_delta * stationary
        denominator = (stationary_velocity != 0).to(foot_delta).sum(
            dim=1
        ) + 1e-4
        velocity_world_update = reset_velocity_world - (
            stationary_velocity.sum(dim=1) / denominator
        )
        velocity_update = torch.matmul(
            current_root_matrix.transpose(-1, -2),
            velocity_world_update.unsqueeze(-1),
        ).squeeze(-1).reshape(1, 1, 3)

        feet_velocity = (
            world_feet - prev_world_feet.reshape(1, 4, 3)
        ) * 30.0 * previous
        feet_feature = (
            feet_velocity * pred_contact.reshape(1, 4, 1)
        ).reshape(1, 1, 12)
        encoder_context = h_enc_out[-1:].transpose(0, 1)
        motion_context = torch.cat(
            (encoder_context, pred_kp3d.reshape(1, 1, 51)), dim=-1
        )
        refiner_context = torch.cat((motion_context, feet_feature), dim=-1)
        (delta_root, delta_velocity), _, (h_refiner_out, c_refiner_out) = (
            self.refiner(
                refiner_context,
                [pred_root.reshape(1, 1, 6), velocity_update],
                (h_refiner_in, c_refiner_in),
            )
        )
        refined_root = pred_root.reshape(1, 1, 6) + delta_root
        refined_velocity = velocity_update + delta_velocity
        previous_refined_matrix = rotation_6d_to_matrix(
            prev_refined_root
        ).reshape(1, 3, 3)
        refined_velocity_world = torch.matmul(
            previous_refined_matrix,
            refined_velocity.reshape(1, 3, 1),
        ).squeeze(-1)
        refined_translation = (
            prev_refined_translation.reshape(1, 3) + refined_velocity_world
        ).reshape(1, 1, 3)
        refined_root_matrix = rotation_6d_to_matrix(refined_root).reshape(1, 3, 3)
        refined_body_rotation = torch.matmul(
            refined_root_matrix, camera_root_matrix.transpose(-1, -2)
        )
        vertices_world = torch.matmul(
            refined_body_rotation.unsqueeze(1), vertices.unsqueeze(-1)
        ).squeeze(-1) + refined_translation.reshape(1, 1, 3)
        joints_world = torch.matmul(
            refined_body_rotation.unsqueeze(1), joints.unsqueeze(-1)
        ).squeeze(-1) + refined_translation.reshape(1, 1, 3)

        return (
            vertices_world.reshape(1, 1, 6890, 3),
            joints_world.reshape(1, 1, 31, 3),
            refined_root,
            refined_velocity,
            refined_translation,
            pred_root.reshape(1, 1, 6),
            unrefined_translation.reshape(1, 1, 3),
            body_feet.reshape(1, 1, 4, 3),
            world_feet.reshape(1, 1, 4, 3),
            h_refiner_out,
            c_refiner_out,
        )


def replace_package(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def relative_error(reference: np.ndarray, actual: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1e-6)
    return float(np.max(np.abs(reference - actual))) / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wham-repo", required=True, type=Path)
    parser.add_argument("--wham-checkpoint", required=True, type=Path)
    parser.add_argument("--hmr2s-checkpoint", required=True, type=Path)
    parser.add_argument("--wham-joint-regressor", required=True, type=Path)
    parser.add_argument("--feet-regressor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if sha256(args.hmr2s_checkpoint) != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError("HMR2.0-S checkpoint checksum mismatch")
    wham_regressor = torch.from_numpy(np.load(args.wham_joint_regressor)).float()
    feet_regressor = torch.from_numpy(np.load(args.feet_regressor)).float()
    if tuple(wham_regressor.shape) != (31, 6890):
        raise RuntimeError(f"Unexpected WHAM regressor shape {wham_regressor.shape}")
    if tuple(feet_regressor.shape) != (4, 6890):
        raise RuntimeError(f"Unexpected feet regressor shape {feet_regressor.shape}")

    network = load_network(args.wham_repo.resolve(), args.wham_checkpoint.resolve())
    wrapper = WHAMWorldStep(
        network.trajectory_refiner,
        checkpoint_smpl_buffers(args.hmr2s_checkpoint),
        wham_regressor,
        feet_regressor,
    ).cpu().eval()
    identity_pose = torch.zeros(1, 1, 144)
    for joint in range(24):
        identity_pose[0, 0, joint * 6] = 1
        identity_pose[0, 0, joint * 6 + 4] = 1
    identity_root = identity_pose[:, :, :6].clone()
    torch.manual_seed(20260914)
    example = (
        identity_pose + torch.randn_like(identity_pose) * 0.02,
        torch.randn(1, 1, 10) * 0.03,
        torch.randn(1, 1, 4),
        identity_root + torch.randn_like(identity_root) * 0.02,
        torch.randn(1, 1, 3) * 0.01,
        torch.randn(1, 1, 51) * 0.1,
        torch.randn(3, 1, 512) * 0.1,
        identity_root,
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 4, 3),
        torch.zeros(1, 1, 4, 3),
        identity_root,
        torch.zeros(1, 1, 3),
        torch.zeros(2, 1, 512),
        torch.zeros(2, 1, 512),
        torch.zeros(1, 1, 1),
    )
    with torch.no_grad():
        references = [value.numpy() for value in wrapper(*example)]
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    input_names = (
        "pred_pose",
        "pred_shape",
        "pred_contact",
        "pred_root",
        "pred_vel",
        "pred_kp3d",
        "h_enc_out",
        "prev_unrefined_root",
        "prev_unrefined_translation",
        "prev_body_feet",
        "prev_world_feet",
        "prev_refined_root",
        "prev_refined_translation",
        "h_refiner_in",
        "c_refiner_in",
        "has_previous",
    )
    output_names = (
        "vertices_world",
        "joints_world",
        "refined_root",
        "refined_velocity",
        "refined_translation",
        "unrefined_root",
        "unrefined_translation",
        "body_feet",
        "world_feet",
        "h_refiner_out",
        "c_refiner_out",
    )
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=name, shape=value.shape)
            for name, value in zip(input_names, example)
        ],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "WHAM-iOS; released WHAM/HMR2.0-S assets"
    converted.license = "Use subject to upstream WHAM, HMR2.0-S, and SMPL terms"
    converted.short_description = "Frozen causal WHAM SMPL and world-refiner step"
    converted.user_defined_metadata.update(
        {
            "operation": "conversion_only_no_training",
            "wham_checkpoint_sha256": sha256(args.wham_checkpoint),
            "hmr2s_checkpoint_sha256": sha256(args.hmr2s_checkpoint),
            "state_contract": "explicit causal refiner and rollout state",
        }
    )
    output = args.output.resolve()
    replace_package(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))

    errors: dict[str, float] = {}
    if platform.system() == "Darwin":
        compiled = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
        payload = {
            name: value.numpy().astype(np.float32)
            for name, value in zip(input_names, example)
        }
        prediction = compiled.predict(payload)
        for name, reference in zip(output_names, references):
            actual = np.asarray(prediction[name])
            error = relative_error(reference, actual)
            errors[name] = error
            if not np.isfinite(actual).all() or error > 0.03:
                raise RuntimeError(
                    f"Core ML world-step parity failed for {name}: {error}"
                )

    report = {
        "schema_version": 1,
        "operation": "conversion_only_no_training",
        "wham_checkpoint_sha256": sha256(args.wham_checkpoint),
        "hmr2s_checkpoint_sha256": sha256(args.hmr2s_checkpoint),
        "output": str(output),
        "coreml_relative_max_errors": errors,
    }
    report_path = args.report or output.parent / "wham_world_step_export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
