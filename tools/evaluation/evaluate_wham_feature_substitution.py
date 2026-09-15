#!/usr/bin/env python3
"""A/B-test official HMR2 and distilled FastViT features inside WHAM.

The official parsed 3DPW file supplies detections, HMR2 features, HMR2 neural
initialization, and ground-truth SMPL rotations. Raw registered 3DPW images are
still required to compute FastViT features on the exact same frames.

This intentionally evaluates the recurrent WHAM core without SMPL assets. It
reports rotation-space diagnostics, not the paper's MPJPE/PVE measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

ARCHITECTURE = "fastvit_sa24_spatial_hmr2_v2"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.astype(np.float64, copy=False), q))


def summarize(values: np.ndarray) -> dict[str, float | int]:
    flat = values.astype(np.float64, copy=False).reshape(-1)
    return {
        "samples": int(flat.size),
        "mean": float(flat.mean()),
        "p50": percentile(flat, 50),
        "p95": percentile(flat, 95),
        "max": float(flat.max()),
    }


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    small = angles.abs() < 1e-6
    scale = torch.empty_like(angles)
    scale[~small] = torch.sin(half_angles[~small]) / angles[~small]
    scale[small] = 0.5 - angles[small].square() / 48.0
    quaternion = torch.cat((torch.cos(half_angles), axis_angle * scale), dim=-1)
    real = quaternion[..., :1]
    imaginary = quaternion[..., 1:]
    two_s = 2.0 / quaternion.square().sum(-1)
    outer = imaginary.unsqueeze(-1) * imaginary.unsqueeze(-2)
    cross = torch.zeros_like(outer)
    x, y, z = imaginary.unbind(-1)
    cross[..., 0, 1] = -z
    cross[..., 0, 2] = y
    cross[..., 1, 0] = z
    cross[..., 1, 2] = -x
    cross[..., 2, 0] = -y
    cross[..., 2, 1] = x
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + two_s[..., None, None] * (
        outer - imaginary.square().sum(-1)[..., None, None] * identity
        + real[..., None] * cross
    )


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    first, second = rotation[..., :3], rotation[..., 3:]
    first = F.normalize(first, dim=-1)
    second = F.normalize(
        second - (first * second).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


def geodesic_degrees_from_matrices(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cosine))


def pose_error_against_axis_angle(
    prediction_6d: torch.Tensor, target_axis_angle: torch.Tensor
) -> torch.Tensor:
    prediction = rotation_6d_to_matrix(prediction_6d.reshape(-1, 24, 6))
    target = axis_angle_to_matrix(target_axis_angle.reshape(-1, 24, 3))
    return geodesic_degrees_from_matrices(prediction, target)


def pose_drift(first_6d: torch.Tensor, second_6d: torch.Tensor) -> torch.Tensor:
    first = rotation_6d_to_matrix(first_6d.reshape(-1, 24, 6))
    second = rotation_6d_to_matrix(second_6d.reshape(-1, 24, 6))
    return geodesic_degrees_from_matrices(first, second)


class FastViTHMR2Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "fastvit_sa24", pretrained=False, num_classes=0
        )
        channels = self.backbone.num_features
        self.spatial_head = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 3)),
            nn.Flatten(),
            nn.LayerNorm(128 * 4 * 3),
            nn.Linear(128 * 4 * 3, 1024),
        )
        self.register_buffer("target_mean", torch.zeros(1024))
        self.register_buffer("target_std", torch.ones(1024))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(image[:, :, :, 32:-32])
        normalized = self.spatial_head(features)
        return normalized * self.target_std + self.target_mean


def load_student(
    checkpoint_path: Path, device: torch.device
) -> tuple[FastViTHMR2Student, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE:
        raise RuntimeError(
            f"Expected {ARCHITECTURE!r}, found {architecture!r} in {checkpoint_path}"
        )
    model = FastViTHMR2Student().eval()
    model.load_state_dict(checkpoint["student_state_dict"], strict=True)
    training_summary = {
        "architecture": architecture,
        "epoch": checkpoint.get("epoch"),
        "stage": checkpoint.get("stage"),
        "accepted": checkpoint.get("accepted"),
        "acceptance_state": checkpoint.get("acceptance_state"),
        "validation": checkpoint.get("validation"),
        "phase3_validation": checkpoint.get("phase3_validation"),
        "phase3_metadata": checkpoint.get("phase3_metadata"),
    }
    return model.to(device), training_summary


def load_wham_core(
    wham_repo: Path, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    sys.path.insert(0, str(wham_repo))
    from lib.models.wham import Network  # pylint: disable=import-outside-toplevel

    network = Network(
        smpl=None,
        pose_dr=0.1,
        d_embed=512,
        n_layers=3,
        d_feat=1024,
        rnn_type="LSTM",
    ).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith("smpl.")
    }
    missing, unexpected = network.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"WHAM checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return network.to(device)


def load_process_image(wham_repo: Path) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    path = wham_repo / "lib/models/preproc/backbone/utils.py"
    spec = importlib.util.spec_from_file_location("_wham_crop_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import WHAM crop utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_image


def locate_image_root(root: Path) -> Path:
    candidates = (
        root / "imageFiles" / "imageFiles",
        root / "imageFiles",
        root / "3DPW" / "imageFiles" / "imageFiles",
        root / "3DPW" / "imageFiles",
        root,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for sequence_dir in candidate.iterdir():
            if sequence_dir.is_dir() and next(
                sequence_dir.glob("image_*.jpg"), None
            ) is not None:
                return candidate
    raise FileNotFoundError(
        "Could not find a 3DPW imageFiles directory with sequence folders "
        f"containing image_*.jpg below {root}"
    )


def sequence_image_paths(
    image_root: Path, video_id: str, frame_ids: np.ndarray
) -> list[Path]:
    sequence_name = video_id.rsplit("_", 1)[0]
    sequence_dir = image_root / sequence_name
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Missing 3DPW sequence directory: {sequence_dir}")
    all_images = sorted(sequence_dir.glob("image_*.jpg"))
    paths: list[Path] = []
    for frame_id in frame_ids.astype(np.int64):
        direct = sequence_dir / f"image_{int(frame_id):05d}.jpg"
        if direct.is_file():
            paths.append(direct)
        elif 0 <= int(frame_id) < len(all_images):
            paths.append(all_images[int(frame_id)])
        else:
            raise FileNotFoundError(
                f"No frame {int(frame_id)} for {video_id} in {sequence_dir}"
            )
    return paths


def normalized_wham_input(
    keypoints: np.ndarray, bbox: np.ndarray, resolution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = bbox[:, :2].astype(np.float32)
    side = bbox[:, 2:3].astype(np.float32)
    coordinates = 2.0 * (keypoints[..., :2] - center[:, None, :]) / side[:, None, :]
    width, height = float(resolution[0]), float(resolution[1])
    longest = max(width, height)
    normalized_center = 2.0 * center / longest - np.asarray(
        [width / longest, height / longest], dtype=np.float32
    )
    location = np.concatenate((normalized_center, side / longest), axis=-1)
    x = np.concatenate((coordinates.reshape(len(keypoints), 34), location), axis=-1)
    mask = keypoints[..., 2] < 0.3
    return x.astype(np.float32), mask


def root_center_coco(joints: np.ndarray) -> np.ndarray:
    joints = joints[..., :17, :3].astype(np.float32, copy=True)
    pelvis = joints[..., [12, 11], :].mean(axis=-2, keepdims=True)
    return joints - pelvis


@torch.inference_mode()
def extract_student_features(
    student: FastViTHMR2Student,
    image_paths: list[Path],
    bbox: np.ndarray,
    process_image: Callable[..., tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    outputs: list[torch.Tensor] = []
    elapsed = 0.0
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="FastViT features",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        tensors: list[torch.Tensor] = []
        for path, box in zip(
            image_paths[start : start + batch_size],
            bbox[start : start + batch_size],
        ):
            with Image.open(path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            normalized, _ = process_image(
                rgb,
                box[:2],
                float(box[2]) / 200.0,
                256,
                256,
            )
            tensors.append(
                torch.from_numpy(np.asarray(normalized, dtype=np.float32))
            )
        images = torch.stack(tensors).to(device, non_blocking=True)
        started = time.perf_counter()
        features = student(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        outputs.append(features.float().cpu())
    return torch.cat(outputs), elapsed


@torch.inference_mode()
def run_wham_core(
    network: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
    init_kp: torch.Tensor,
    init_pose: torch.Tensor,
    init_root: torch.Tensor,
    cam_angvel: torch.Tensor,
) -> dict[str, torch.Tensor]:
    processed = network.preprocess(x.clone(), mask)
    pred_kp3d, motion_context = network.motion_encoder(processed, init_kp)
    pred_root, pred_vel = network.trajectory_decoder(
        motion_context, init_root, cam_angvel
    )
    integrated = network.integrator(motion_context, features)
    pred_pose, pred_shape, pred_cam, pred_contact = network.motion_decoder(
        integrated, init_pose
    )
    return {
        "pose": pred_pose,
        "kp3d": pred_kp3d,
        "root": pred_root[:, 1:],
        "velocity": pred_vel,
        "shape": pred_shape,
        "camera": pred_cam,
        "contact": pred_contact,
    }


def rotation_summary(errors: np.ndarray) -> dict[str, Any]:
    return {
        "all_joints_deg": summarize(errors),
        "root_deg": summarize(errors[:, 0]),
        "body_deg": summarize(errors[:, 1:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--parsed-3dpw", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=10)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--max-pose-degradation-deg", type=float, default=1.0)
    parser.add_argument("--max-relative-pose-degradation", type=float, default=0.10)
    parser.add_argument("--max-teacher-drift-deg", type=float, default=5.0)
    parser.add_argument(
        "--acceptance-policy-label",
        default="predeclared_validation_policy",
        help="Human-readable provenance label stored with the thresholds",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-output", type=Path)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()

    required_files = (
        args.parsed_3dpw,
        args.student_checkpoint,
        args.wham_checkpoint,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))
    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if wham_commit != WHAM_COMMIT:
        raise RuntimeError(
            f"Expected WHAM commit {WHAM_COMMIT}, found {wham_commit}"
        )
    image_root = locate_image_root(args.three_dpw_root.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; raw 3DPW images: {image_root}", flush=True)

    labels = joblib.load(args.parsed_3dpw)
    student, student_training = load_student(args.student_checkpoint, device)
    student_identity = {
        "epoch": student_training["epoch"],
        "stage": student_training["stage"],
        "training_accepted": student_training["accepted"],
        "acceptance_state": student_training["acceptance_state"],
        "validation_raw_cosine": (student_training["validation"] or {}).get(
            "raw_cosine_mean"
        ),
        "phase3_validation_accepted": (
            student_training["phase3_validation"] or {}
        ).get("accepted_on_validation"),
    }
    print(f"Student checkpoint: {json.dumps(student_identity)}", flush=True)
    network = load_wham_core(args.wham_repo.resolve(), args.wham_checkpoint, device)
    process_image = load_process_image(args.wham_repo.resolve())

    matching: list[int] = []
    for index, raw_video_id in enumerate(labels["vid"]):
        video_id = str(raw_video_id)
        if (image_root / video_id.rsplit("_", 1)[0]).is_dir():
            matching.append(index)
    if not matching:
        raise RuntimeError(
            "No parsed 3DPW sequence names matched the supplied imageFiles directory"
        )
    if args.sequences <= 0:
        selected = matching
    else:
        sample_count = min(args.sequences, len(matching))
        positions = np.linspace(
            0, len(matching) - 1, num=sample_count, dtype=np.int64
        )
        selected = [matching[int(position)] for position in positions]

    all_teacher_features: list[torch.Tensor] = []
    all_student_features: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_teacher_pose: list[torch.Tensor] = []
    all_student_pose: list[torch.Tensor] = []
    per_sequence: list[dict[str, Any]] = []
    feature_seconds = 0.0
    wham_seconds = {"teacher": 0.0, "student": 0.0}

    for index in selected:
        video_id = str(labels["vid"][index])
        keypoints = to_numpy(labels["kp2d"][index])
        bbox = to_numpy(labels["bbox"][index])
        frame_ids = to_numpy(labels["frame_id"][index])
        resolution = to_numpy(labels["res"][index][0])
        available = min(
            len(keypoints) - 1,
            len(bbox) - 1,
            len(frame_ids) - 1,
            len(labels["features"][index]) - 1,
            len(labels["pose"][index]) - 1,
        )
        if args.frames > 0:
            available = min(args.frames, available)
        if available <= 0:
            continue
        frame_slice = slice(1, available + 1)
        paths = sequence_image_paths(
            image_root, video_id, frame_ids[frame_slice]
        )
        student_features, seconds = extract_student_features(
            student,
            paths,
            bbox[frame_slice],
            process_image,
            args.student_batch_size,
            device,
        )
        feature_seconds += seconds
        teacher_features = torch.from_numpy(
            to_numpy(labels["features"][index])[frame_slice].astype(np.float32)
        )
        x_all, mask_all = normalized_wham_input(keypoints, bbox, resolution)
        x = torch.from_numpy(x_all[frame_slice]).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask_all[frame_slice]).unsqueeze(0).to(device)
        init_kp3d = root_center_coco(to_numpy(labels["init_kp3d"][index])[0])
        init_kp = np.concatenate((init_kp3d.reshape(-1), x_all[0]), axis=0)
        init_pose_axis_angle = torch.from_numpy(
            to_numpy(labels["init_pose"][index])[0].astype(np.float32)
        ).reshape(24, 3)
        init_pose = matrix_to_rotation_6d(
            axis_angle_to_matrix(init_pose_axis_angle)
        ).reshape(1, 1, 24, 6)
        target = torch.from_numpy(
            to_numpy(labels["pose"][index])[frame_slice].astype(np.float32)
        ).reshape(available, 24, 3)
        target_root = matrix_to_rotation_6d(
            axis_angle_to_matrix(target[0, 0])
        ).reshape(1, 1, 6)
        common = {
            "network": network,
            "x": x,
            "mask": mask,
            "init_kp": torch.from_numpy(init_kp.astype(np.float32))
            .reshape(1, 1, 88)
            .to(device),
            "init_pose": init_pose.to(device),
            "init_root": target_root.to(device),
            "cam_angvel": torch.zeros(1, available, 6, device=device),
        }

        started = time.perf_counter()
        teacher_output = run_wham_core(
            features=teacher_features.unsqueeze(0).to(device), **common
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wham_seconds["teacher"] += time.perf_counter() - started
        started = time.perf_counter()
        student_output = run_wham_core(
            features=student_features.unsqueeze(0).to(device), **common
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wham_seconds["student"] += time.perf_counter() - started

        teacher_pose = teacher_output["pose"].squeeze(0).cpu()
        student_pose = student_output["pose"].squeeze(0).cpu()
        target_cpu = target.cpu()
        teacher_error = pose_error_against_axis_angle(teacher_pose, target_cpu)
        student_error = pose_error_against_axis_angle(student_pose, target_cpu)
        drift = pose_drift(student_pose, teacher_pose)
        cosine = F.cosine_similarity(student_features, teacher_features, dim=-1)
        per_sequence.append(
            {
                "sequence": video_id,
                "frames": available,
                "feature_cosine_mean": float(cosine.mean()),
                "teacher_pose_error_deg": float(teacher_error.mean()),
                "student_pose_error_deg": float(student_error.mean()),
                "pose_degradation_deg": float(student_error.mean() - teacher_error.mean()),
                "student_teacher_pose_drift_deg": float(drift.mean()),
                "student_teacher_root_drift_deg": float(drift[:, 0].mean()),
            }
        )
        print(json.dumps(per_sequence[-1]), flush=True)

        all_teacher_features.append(teacher_features)
        all_student_features.append(student_features)
        all_targets.append(target_cpu)
        all_teacher_pose.append(teacher_pose)
        all_student_pose.append(student_pose)

    if not per_sequence:
        raise RuntimeError("No non-empty 3DPW person tracks were evaluated")

    teacher_features = torch.cat(all_teacher_features)
    student_features = torch.cat(all_student_features)
    target = torch.cat(all_targets)
    teacher_pose = torch.cat(all_teacher_pose)
    student_pose = torch.cat(all_student_pose)
    teacher_error = pose_error_against_axis_angle(teacher_pose, target).numpy()
    student_error = pose_error_against_axis_angle(student_pose, target).numpy()
    drift = pose_drift(student_pose, teacher_pose).numpy()
    cosine = F.cosine_similarity(student_features, teacher_features, dim=-1).numpy()
    feature_rmse = float(torch.sqrt(F.mse_loss(student_features, teacher_features)))
    teacher_mean = float(teacher_error.mean())
    student_mean = float(student_error.mean())
    degradation = student_mean - teacher_mean
    relative_degradation = degradation / max(teacher_mean, 1e-8)
    drift_mean = float(drift.mean())
    gates = {
        "absolute_pose_degradation": degradation <= args.max_pose_degradation_deg,
        "relative_pose_degradation": relative_degradation
        <= args.max_relative_pose_degradation,
        "student_teacher_pose_drift": drift_mean <= args.max_teacher_drift_deg,
    }
    accepted = all(gates.values())
    report = {
        "schema_version": 1,
        "accepted_for_device_diagnostic": accepted,
        "gates": gates,
        "thresholds": {
            "acceptance_policy_label": args.acceptance_policy_label,
            "max_pose_degradation_deg": args.max_pose_degradation_deg,
            "max_relative_pose_degradation": args.max_relative_pose_degradation,
            "max_student_teacher_pose_drift_deg": args.max_teacher_drift_deg,
        },
        "scope": {
            "dataset": "registered 3DPW test images + official WHAM parsed ViT labels",
            "person_tracks": len(per_sequence),
            "frames": len(target),
            "flip_evaluation": False,
            "smpl_forward": False,
            "metric_note": (
                "Rotation-space downstream diagnostic; not the paper's "
                "PA-MPJPE/MPJPE/PVE evaluation."
            ),
        },
        "provenance": {
            "wham_commit": wham_commit,
            "wham_checkpoint_sha256": sha256_file(args.wham_checkpoint),
            "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
            "student_training": student_training,
            "parsed_3dpw_sha256": sha256_file(args.parsed_3dpw),
        },
        "feature": {
            "cosine": summarize(cosine),
            "rmse": feature_rmse,
        },
        "teacher_wham_vs_ground_truth": rotation_summary(teacher_error),
        "student_wham_vs_ground_truth": rotation_summary(student_error),
        "student_minus_teacher": {
            "pose_error_degradation_deg": degradation,
            "relative_pose_error_degradation": relative_degradation,
            "pose_drift": rotation_summary(drift),
        },
        "gpu_work_seconds": {
            "student_feature_extraction": feature_seconds,
            "wham_teacher_features": wham_seconds["teacher"],
            "wham_student_features": wham_seconds["student"],
        },
        "per_sequence": per_sequence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    per_sequence_path = args.per_sequence_output or args.output.with_suffix(".csv")
    with per_sequence_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sequence[0]))
        writer.writeheader()
        writer.writerows(per_sequence)
    compact = {
        "accepted_for_device_diagnostic": accepted,
        "person_tracks": len(per_sequence),
        "frames": len(target),
        "feature_cosine_mean": report["feature"]["cosine"]["mean"],
        "teacher_pose_error_deg": teacher_mean,
        "student_pose_error_deg": student_mean,
        "pose_error_degradation_deg": degradation,
        "relative_pose_error_degradation": relative_degradation,
        "student_teacher_pose_drift_deg": drift_mean,
        "student_epoch": student_training["epoch"],
        "student_training_accepted": student_training["accepted"],
    }
    print(json.dumps(compact, indent=2))
    print(f"Report: {args.output}")
    print(f"Per-sequence metrics: {per_sequence_path}")
    if args.fail_on_reject and not accepted:
        raise SystemExit(
            "REJECTED: FastViT degrades the downstream WHAM rotation diagnostic"
        )


if __name__ == "__main__":
    main()
