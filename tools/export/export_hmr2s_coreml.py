#!/usr/bin/env python3
"""Export the released, frozen HMR2.0-S front end for the iPhone benchmark.

Two packages are produced:

* HMR2SFrontend.mlpackage: 256x256 RGB crop -> WHAM token, SMPL pose,
  shape, and camera.
* HMR2SSMPLInit.mlpackage: first-frame pose and shape -> the 17 root-centred
  3D joints consumed by WHAM_I.

No parameter is trained or modified.  The script checks the official
checkpoint SHA-256, strictly loads the released state dictionary, converts the
graphs, and performs Core ML/PyTorch numerical parity checks on macOS.
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

from hmr2s_frozen import (
    HMR2S_CHECKPOINT_SHA256,
    HMR2S_COMMIT,
    HMR2S_REPOSITORY,
    FrozenHMR2S,
    checkpoint_smpl_buffers,
)
from PIL import Image
from smplx.lbs import lbs
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ByteImageHMR2S(nn.Module):
    """Expose an image-friendly 0...255 RGB interface with exact normalization."""

    def __init__(self, model: FrozenHMR2S) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def forward(
        self, image_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (image_input / 255.0 - self.mean) / self.std
        token, pose, betas, camera = self.model(normalized)
        return (
            token.reshape(1, 1, 1024),
            pose.reshape(1, 1, 24, 6),
            betas.reshape(1, 1, 10),
            camera.reshape(1, 1, 3),
        )


def wham_rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """WHAM convention: its six numbers are the first two matrix rows."""

    rows = rotation.reshape(-1, 2, 3)
    first = torch.nn.functional.normalize(rows[:, 0], dim=-1)
    second_raw = rows[:, 1]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


class HMR2SSMPLInitializer(nn.Module):
    """The official frozen neutral SMPL body used only on the first frame."""

    def __init__(
        self,
        buffers: dict[str, torch.Tensor],
        wham_joint_regressor: torch.Tensor,
    ) -> None:
        super().__init__()
        for name, value in buffers.items():
            self.register_buffer(name, value)
        self.register_buffer("wham_joint_regressor", wham_joint_regressor.float())

    def forward(
        self, pose_6d: torch.Tensor, betas: torch.Tensor
    ) -> torch.Tensor:
        matrices = wham_rotation_6d_to_matrix(pose_6d).reshape(1, 24, 3, 3)
        vertices, _ = lbs(
            betas.reshape(1, 10),
            matrices,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )
        joints = torch.einsum("jv,bvc->bjc", self.wham_joint_regressor, vertices)
        joints = joints[:, :17]
        pelvis = joints[:, [12, 11]].mean(dim=1, keepdim=True)
        return (joints - pelvis).reshape(1, 1, 51)


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


def convert_frontend(
    wrapper: ByteImageHMR2S,
    example: torch.Tensor,
    output: Path,
) -> tuple[list[np.ndarray], dict[str, float]]:
    with torch.no_grad():
        torch_outputs = [value.numpy() for value in wrapper(example)]
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image_input",
                shape=example.shape,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="image_token"),
            ct.TensorType(name="pose_6d"),
            ct.TensorType(name="betas"),
            ct.TensorType(name="camera"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    model.author = "NTT / WHAM-iOS packaging"
    model.license = "Use subject to the HMR2.0-S authors' release terms"
    model.short_description = (
        "Frozen released HMR2.0-S front end; no project-specific training"
    )
    model.user_defined_metadata.update(
        {
            "source_repository": HMR2S_REPOSITORY,
            "source_commit": HMR2S_COMMIT,
            "checkpoint_sha256": HMR2S_CHECKPOINT_SHA256,
            "training_performed_by_this_project": "false",
            "wham_token_dimension": "1024",
            "crop_contract": "256x256 RGB; center 256x192 is consumed",
        }
    )
    replace_package(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))

    errors: dict[str, float] = {}
    if platform.system() == "Darwin":
        compiled = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
        rgb = example[0].permute(1, 2, 0).byte().numpy()
        prediction = compiled.predict({"image_input": Image.fromarray(rgb, "RGB")})
        for name, reference in zip(
            ("image_token", "pose_6d", "betas", "camera"), torch_outputs
        ):
            actual = np.asarray(prediction[name])
            error = relative_error(reference, actual)
            errors[name] = error
            if not np.isfinite(error) or error > 0.02:
                raise RuntimeError(
                    f"Core ML front-end parity failed for {name}: relative max error {error}"
                )
    return torch_outputs, errors


def convert_initializer(
    wrapper: HMR2SSMPLInitializer,
    pose: torch.Tensor,
    betas: torch.Tensor,
    output: Path,
) -> dict[str, float]:
    with torch.no_grad():
        reference = wrapper(pose, betas).numpy()
    traced = torch.jit.trace(wrapper, (pose, betas), check_trace=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pose_6d", shape=pose.shape),
            ct.TensorType(name="betas", shape=betas.shape),
        ],
        outputs=[ct.TensorType(name="init_kp3d")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    model.author = "NTT / WHAM-iOS packaging"
    model.license = "Use subject to the SMPL and HMR2.0-S model terms"
    model.short_description = "Frozen HMR2.0-S neutral-SMPL first-frame initializer"
    model.user_defined_metadata.update(
        {
            "checkpoint_sha256": HMR2S_CHECKPOINT_SHA256,
            "training_performed_by_this_project": "false",
            "output_joint_convention": "WHAM first 17 joints, COCO pelvis-centred",
            "frequency": "first frame of each track only",
        }
    )
    replace_package(output)
    model.save(str(output))

    errors: dict[str, float] = {}
    if platform.system() == "Darwin":
        compiled = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
        prediction = compiled.predict(
            {
                "pose_6d": pose.numpy().astype(np.float32),
                "betas": betas.numpy().astype(np.float32),
            }
        )
        actual = np.asarray(prediction["init_kp3d"])
        error = relative_error(reference, actual)
        errors["init_kp3d"] = error
        if not np.isfinite(error) or error > 0.02:
            raise RuntimeError(
                f"Core ML SMPL initializer parity failed: relative max error {error}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wham-joint-regressor", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    actual_sha = sha256(args.checkpoint)
    if actual_sha != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError(
            "HMR2.0-S checkpoint SHA-256 mismatch: "
            f"expected {HMR2S_CHECKPOINT_SHA256}, got {actual_sha}"
        )
    regressor = torch.from_numpy(np.load(args.wham_joint_regressor)).float()
    if tuple(regressor.shape) != (31, 6890):
        raise RuntimeError(
            f"Unexpected WHAM joint regressor shape: {tuple(regressor.shape)}"
        )

    torch.manual_seed(20260914)
    frozen = FrozenHMR2S(args.repository, args.checkpoint).cpu().eval()
    frontend = ByteImageHMR2S(frozen).cpu().eval()
    example = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)

    output_directory = args.output_directory.resolve()
    frontend_path = output_directory / "HMR2SFrontend.mlpackage"
    initializer_path = output_directory / "HMR2SSMPLInit.mlpackage"
    frontend_outputs, frontend_errors = convert_frontend(
        frontend, example, frontend_path
    )

    initializer = HMR2SSMPLInitializer(
        checkpoint_smpl_buffers(args.checkpoint), regressor
    ).cpu().eval()
    pose = torch.from_numpy(frontend_outputs[1]).float()
    betas = torch.from_numpy(frontend_outputs[2]).float()
    initializer_errors = convert_initializer(
        initializer, pose, betas, initializer_path
    )

    report = {
        "schema_version": 1,
        "operation": "conversion_only_no_training",
        "source_repository": HMR2S_REPOSITORY,
        "source_commit": HMR2S_COMMIT,
        "checkpoint_sha256": actual_sha,
        "parameter_count": sum(parameter.numel() for parameter in frozen.parameters()),
        "packages": {
            "frontend": str(frontend_path),
            "first_frame_smpl_initializer": str(initializer_path),
        },
        "coreml_relative_max_errors": {**frontend_errors, **initializer_errors},
    }
    report_path = args.report or output_directory / "hmr2s_coreml_export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
