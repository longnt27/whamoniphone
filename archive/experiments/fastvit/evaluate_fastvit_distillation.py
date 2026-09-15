#!/usr/bin/env python3
"""Compare the mobile FastViT token with the original HMR2 teacher token."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from benchmark_coreml_pipeline import detect_person, preprocess_observation
from PIL import Image
from torch import nn


class HMR2TokenEncoder(nn.Module):
    """Only the HMR2 modules used by `teacher(image, encode=True)`."""

    def __init__(self, wham_repo: Path) -> None:
        super().__init__()
        backbone_root = wham_repo / "lib/models/preproc/backbone"
        required = (backbone_root / "pose_transformer.py", backbone_root / "vit.py")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"Incomplete HMR2 backbone at {backbone_root}")
        package_name = "_wham_hmr2_backbone"
        package = types.ModuleType(package_name)
        package.__path__ = [str(backbone_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        pose_transformer = importlib.import_module(f"{package_name}.pose_transformer")
        vit_module = importlib.import_module(f"{package_name}.vit")

        self.backbone = vit_module.vit()
        self.transformer = pose_transformer.TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=1280,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(image[:, :, :, 32:-32])
        context = feature_map.flatten(2).transpose(1, 2)
        token = torch.zeros(image.shape[0], 1, 1, device=image.device)
        return self.transformer(token, context=context).squeeze(1)


def subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


def load_teacher(wham_repo: Path, checkpoint_path: Path) -> HMR2TokenEncoder:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    model = HMR2TokenEncoder(wham_repo).eval()
    model.backbone.load_state_dict(subset(state, "backbone."), strict=True)
    model.transformer.load_state_dict(subset(state, "smpl_head.transformer."), strict=True)
    return model


def normalized_tensor(crop: Image.Image) -> torch.Tensor:
    pixels = np.asarray(crop, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)


def rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    pair = rotation.reshape(-1, 24, 2, 3)
    first = pair[..., 0, :]
    second = pair[..., 1, :]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def pose_geodesic_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_matrix = rotation_6d_to_matrix(first)
    second_matrix = rotation_6d_to_matrix(second)
    relative = np.swapaxes(first_matrix, -1, -2) @ second_matrix
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wham-repo", required=True, type=Path)
    parser.add_argument("--hmr2-checkpoint", required=True, type=Path)
    parser.add_argument("--frames", type=Path, default=Path("WhamApp/WhamApp/BenchmarkFrames"))
    parser.add_argument("--models", type=Path, default=Path("WhamApp/WhamApp"))
    parser.add_argument("--pose-head", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Resolve before the teacher switches cwd to the upstream checkout.
    frame_paths = sorted(path.resolve() for path in args.frames.glob("frame_*.jpg"))
    model_dir = args.models.resolve()
    pose_head_path = (
        args.pose_head.resolve()
        if args.pose_head
        else model_dir / "FastViTInitialPose.mlpackage"
    )
    output_path = args.output.resolve() if args.output else None
    if not frame_paths:
        parser.error("No frame_*.jpg files found")

    yolo = ct.models.MLModel(str(model_dir / "yolov8n-pose.mlpackage"))
    student = ct.models.MLModel(str(model_dir / "FastViTNormalized.mlpackage"))
    pose_head = ct.models.MLModel(str(pose_head_path))
    teacher = load_teacher(args.wham_repo.resolve(), args.hmr2_checkpoint.resolve())

    teacher_features: list[np.ndarray] = []
    student_features: list[np.ndarray] = []
    teacher_times: list[float] = []
    student_times: list[float] = []

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        detection = detect_person(yolo, image)
        if detection is None:
            raise RuntimeError(f"No person detected in {frame_path.name}")
        _, _, crop = preprocess_observation(image, detection)
        assert crop is not None

        input_tensor = normalized_tensor(crop)
        started = time.perf_counter_ns()
        with torch.no_grad():
            teacher_feature = teacher(input_tensor).numpy()
        teacher_times.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        student_feature = student.predict({"image_input": crop})["features_1024"]
        student_times.append((time.perf_counter_ns() - started) / 1_000_000)
        teacher_features.append(teacher_feature.reshape(1024))
        student_features.append(student_feature.reshape(1024))

    teacher_array = np.asarray(teacher_features)
    student_array = np.asarray(student_features)
    cosine = np.sum(teacher_array * student_array, axis=-1) / np.maximum(
        np.linalg.norm(teacher_array, axis=-1) * np.linalg.norm(student_array, axis=-1),
        1e-8,
    )
    teacher_pose = np.concatenate(
        [
            pose_head.predict({"image_feature": value.reshape(1, 1, 1024)})[
                "smpl_pose_6d"
            ]
            for value in teacher_array
        ],
        axis=0,
    )
    student_pose = np.concatenate(
        [
            pose_head.predict({"image_feature": value.reshape(1, 1, 1024)})[
                "smpl_pose_6d"
            ]
            for value in student_array
        ],
        axis=0,
    )
    pose_error = pose_geodesic_degrees(student_pose, teacher_pose)

    report = {
        "schema_version": 1,
        "frames": [path.name for path in frame_paths],
        "feature": {
            "cosine_similarity_mean": float(cosine.mean()),
            "cosine_similarity_min": float(cosine.min()),
            "rmse": float(np.sqrt(np.mean((student_array - teacher_array) ** 2))),
            "mean_absolute_error": float(np.mean(np.abs(student_array - teacher_array))),
            "teacher_l2_mean": float(np.linalg.norm(teacher_array, axis=-1).mean()),
            "student_l2_mean": float(np.linalg.norm(student_array, axis=-1).mean()),
        },
        "initial_pose_student_vs_teacher": {
            "joint_rotation_error_deg": float(pose_error.mean()),
            "body_rotation_error_deg": float(pose_error[:, 1:].mean()),
            "root_rotation_error_deg": float(pose_error[:, 0].mean()),
        },
        "latency": {
            "hmr2_teacher_mean_ms": float(np.mean(teacher_times)),
            "fastvit_coreml_mean_ms": float(np.mean(student_times)),
        },
        "note": "Mini-set distillation diagnostic without pose ground truth; teacher output is a reference, not ground truth.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
