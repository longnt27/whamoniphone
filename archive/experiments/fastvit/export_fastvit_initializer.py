#!/usr/bin/env python3
"""Export HMR2's SMPL pose readout for the distilled 1024-D FastViT token.

The FastViT student in this repository was trained against
`teacher(image, encode=True)`, which is HMR2's 1024-D SMPL decoder token. The
official HMR2 `decpose` layer can therefore turn that token into a first-frame
SMPL pose without shipping the 2.7 GB ViT-H backbone.

This model does not produce SMPL joints. A full WHAM initializer still needs a
licensed SMPL body model (or a separately trained 3D-joint head) for the first
51 values of `init_kp`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


class HMR2PoseReadout(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, mean_pose: torch.Tensor) -> None:
        super().__init__()
        self.pose = nn.Linear(1024, 144)
        self.pose.weight.data.copy_(weight)
        self.pose.bias.data.copy_(bias)
        self.register_buffer("mean_pose", mean_pose.reshape(1, 144))

    def forward(self, feature):
        raw_6d = self.pose(feature.reshape(-1, 1024)) + self.mean_pose

        # Match HMR2's rot6d_to_rotmat, then WHAM's matrix_to_rotation_6d.
        pair = raw_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        first = functional.normalize(pair[:, :, 0], dim=-1)
        second_raw = pair[:, :, 1] - (
            (first * pair[:, :, 1]).sum(dim=-1, keepdim=True) * first
        )
        second = functional.normalize(second_raw, dim=-1)
        third = torch.cross(first, second, dim=-1)
        matrix = torch.stack((first, second, third), dim=-1).reshape(-1, 24, 3, 3)
        return matrix[..., :2, :].reshape(-1, 1, 144)


def find_tensor(state: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        keys = [key for key in state if key.endswith(suffix)]
        raise KeyError(f"Expected one checkpoint key ending in {suffix!r}; found {keys}")
    return matches[0].detach().cpu().float()


def load_readout(checkpoint_path: Path) -> HMR2PoseReadout:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    return HMR2PoseReadout(
        find_tensor(state, "smpl_head.decpose.weight"),
        find_tensor(state, "smpl_head.decpose.bias"),
        find_tensor(state, "smpl_head.init_body_pose"),
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmr2-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = load_readout(args.hmr2_checkpoint)
    example = torch.randn(1, 1, 1024)
    traced = torch.jit.trace(model, example, check_trace=True)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image_feature", shape=example.shape)],
        outputs=[ct.TensorType(name="smpl_pose_6d")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "WHAM-iOS; HMR2 pose readout from the official WHAM checkpoint"
    converted.license = "WHAM/HMR2 research assets; see the upstream projects"
    converted.short_description = "First-frame SMPL pose from the distilled HMR2 token"

    output = args.output.resolve()
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))

    compiled = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
    with torch.no_grad():
        expected = model(example).numpy()
    actual = np.asarray(
        compiled.predict({"image_feature": example.numpy()})["smpl_pose_6d"]
    )
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or error > 0.02:
        raise RuntimeError(f"Core ML parity check failed (max error={error})")
    print(f"Saved {output}")
    print(f"Core ML parity: max |pose delta| = {error:.6f}")


if __name__ == "__main__":
    main()
