#!/usr/bin/env python3
"""Evaluate the exported causal models on official parsed WHAM test data.

The official paper metrics require licensed SMPL body assets.  This script does
not substitute a different skeleton: it reports SMPL joint rotation error, an
asset-free diagnostic, and labels it accordingly.  It also performs controlled
initialization and image-feature ablations and records Core ML latency.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import joblib
import numpy as np
import torch


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = torch.from_numpy(axis_angle.astype(np.float32, copy=False))
    angles = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
    axes = vectors / safe_angles
    x, y, z = axes.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
        *vectors.shape[:-1], 3, 3
    )
    identity = torch.eye(3, dtype=vectors.dtype).expand(*vectors.shape[:-1], 3, 3)
    sin = torch.sin(angles)[..., None]
    cos = torch.cos(angles)[..., None]
    matrix = identity + sin * skew + (1.0 - cos) * (skew @ skew)
    return matrix.numpy()


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    pair = rotation.reshape(*rotation.shape[:-1], 2, 3)
    a1, a2 = pair[..., 0, :], pair[..., 1, :]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    projection = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2_raw = a2 - projection
    b2 = b2_raw / np.maximum(np.linalg.norm(b2_raw, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def geodesic_degrees(predicted_6d: np.ndarray, target_axis_angle: np.ndarray) -> np.ndarray:
    predicted = rotation_6d_to_matrix(predicted_6d.reshape(-1, 24, 6))
    target = axis_angle_to_matrix(target_axis_angle.reshape(-1, 24, 3))
    relative = np.swapaxes(predicted, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def normalized_wham_input(kp2d: np.ndarray, bbox: np.ndarray, resolution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce WHAM's 224-square crop normalization without OpenCV."""
    center = bbox[:, :2]
    side = bbox[:, 2:3]
    coordinates = 2.0 * (kp2d[..., :2] - center[:, None, :]) / side[:, None, :]

    width, height = float(resolution[0]), float(resolution[1])
    longest = max(width, height)
    normalized_center = 2.0 * center / longest - np.asarray([width / longest, height / longest])
    normalized_scale = side / longest
    location = np.concatenate((normalized_center, normalized_scale), axis=-1)
    x = np.concatenate((coordinates.reshape(len(kp2d), 34), location), axis=-1)
    missing = (kp2d[..., 2] < 0.3).astype(np.float32)
    return x.astype(np.float32), missing


@dataclass
class SequenceInputs:
    name: str
    x: np.ndarray
    mask: np.ndarray
    features: np.ndarray
    initial_feature: np.ndarray
    init_kp: np.ndarray
    init_pose: np.ndarray
    init_pose_axis_angle: np.ndarray
    target_pose_axis_angle: np.ndarray


def prepare_sequence(dataset: dict[str, Any], index: int, frame_limit: int) -> SequenceInputs:
    kp2d = dataset["kp2d"][index].numpy()
    bbox = dataset["bbox"][index].numpy()
    resolution = dataset["res"][index][0].numpy()
    normalized, missing = normalized_wham_input(kp2d, bbox, resolution)

    frames = min(frame_limit, len(normalized) - 1)
    init_kp3d = dataset["init_kp3d"][index][0, :17].numpy().reshape(-1)
    init_kp = np.concatenate((init_kp3d, normalized[0]), axis=0).reshape(1, 1, 88)
    init_axis_angle = dataset["init_pose"][index][0].numpy()
    init_pose = matrix_to_rotation_6d(axis_angle_to_matrix(init_axis_angle)).reshape(1, 1, 144)

    return SequenceInputs(
        name=str(dataset["vid"][index]),
        x=normalized[1 : frames + 1],
        mask=missing[1 : frames + 1],
        features=dataset["features"][index][1 : frames + 1].numpy().astype(np.float32),
        initial_feature=dataset["features"][index][0].numpy().astype(np.float32).reshape(1, 1, 1024),
        init_kp=init_kp.astype(np.float32),
        init_pose=init_pose.astype(np.float32),
        init_pose_axis_angle=init_axis_angle.astype(np.float32).reshape(1, 24, 3),
        target_pose_axis_angle=dataset["pose"][index][1 : frames + 1].numpy().astype(np.float32),
    )


STATE_NAMES = (
    "h_enc",
    "c_enc",
    "h_traj",
    "c_traj",
    "h_dec",
    "c_dec",
)


def identity_pose() -> np.ndarray:
    pose = np.zeros((1, 1, 144), dtype=np.float32)
    for joint in range(24):
        pose[0, 0, joint * 6] = 1
        pose[0, 0, joint * 6 + 4] = 1
    return pose


def initialize(
    init_model: ct.models.MLModel,
    sequence: SequenceInputs,
    mode: str,
    pose_init_model: ct.models.MLModel | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if mode == "legacy_zero":
        init_kp = np.zeros((1, 1, 88), dtype=np.float32)
        init_pose = np.zeros((1, 1, 144), dtype=np.float32)
        previous_root = np.zeros((1, 1, 6), dtype=np.float32)
    elif mode == "official":
        init_kp = sequence.init_kp
        init_pose = sequence.init_pose
        previous_root = init_pose[..., :6].copy()
    elif mode in ("mobile_neutral", "mobile_feature_pose"):
        # The phone has exact first-frame 2D/crop values, but no SMPL body model
        # for HMR2's root-centered 3D joints.
        init_kp = sequence.init_kp.copy()
        init_kp[..., :51] = 0
        if mode == "mobile_feature_pose":
            if pose_init_model is None:
                raise ValueError("mobile_feature_pose requires --pose-init-model")
            init_pose = pose_init_model.predict(
                {"image_feature": sequence.initial_feature}
            )["smpl_pose_6d"]
        else:
            init_pose = identity_pose()
        previous_root = init_pose[..., :6].copy()
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")
    output = init_model.predict({"init_kp": init_kp, "init_smpl": init_pose})
    previous_kp3d = init_kp[..., :51].copy()
    return {name: output[name] for name in STATE_NAMES}, init_pose, previous_kp3d, previous_root


def run_sequence(
    init_model: ct.models.MLModel,
    step_model: ct.models.MLModel,
    sequence: SequenceInputs,
    *,
    image_model: bool,
    initialization: str,
    image_valid: bool,
    pose_init_model: ct.models.MLModel | None,
) -> tuple[np.ndarray, list[float]]:
    state, previous_pose, previous_kp3d, previous_root = initialize(
        init_model, sequence, initialization, pose_init_model
    )
    camera_angular_velocity = np.zeros((1, 1, 6), dtype=np.float32)
    predictions: list[np.ndarray] = []
    latencies: list[float] = []

    for frame_index, x_step in enumerate(sequence.x):
        model_input = {
            "x_step": x_step.reshape(1, 1, 37),
            "cam_a_step": camera_angular_velocity,
            "prev_kp3d": previous_kp3d,
            "prev_root": previous_root,
            "prev_pose": previous_pose,
            "h_enc_in": state["h_enc"],
            "c_enc_in": state["c_enc"],
            "h_traj_in": state["h_traj"],
            "c_traj_in": state["c_traj"],
            "h_dec_in": state["h_dec"],
            "c_dec_in": state["c_dec"],
        }
        if image_model:
            model_input.update(
                {
                    "keypoint_mask_step": sequence.mask[frame_index].reshape(1, 1, 17),
                    "image_feature_step": (
                        sequence.features[frame_index].reshape(1, 1, 1024)
                        if image_valid
                        else np.zeros((1, 1, 1024), dtype=np.float32)
                    ),
                    "image_feature_valid_step": np.asarray([[[float(image_valid)]]], dtype=np.float32),
                }
            )

        started = time.perf_counter_ns()
        output = step_model.predict(model_input)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        predictions.append(output["pred_pose"].reshape(144))
        previous_kp3d = output["pred_kp3d"]
        previous_root = output["pred_root"]
        previous_pose = output["pred_pose"]
        for stem in STATE_NAMES:
            state[stem] = output[f"{stem}_out"]

    return np.asarray(predictions), latencies


def summarize_variant(predictions: list[np.ndarray], targets: list[np.ndarray], latency: list[float]) -> dict[str, float]:
    predicted = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    error = geodesic_degrees(predicted, target)
    return {
        "frames": int(len(predicted)),
        "pose_rotation_error_deg": float(error.mean()),
        "body_rotation_error_deg": float(error[:, 1:].mean()),
        "root_rotation_error_deg": float(error[:, 0].mean()),
        "latency_mean_ms": float(statistics.fmean(latency)),
        "latency_p50_ms": percentile(latency, 50),
        "latency_p95_ms": percentile(latency, 95),
        "throughput_fps_from_mean": float(1000.0 / statistics.fmean(latency)),
        "nonfinite_values": int((~np.isfinite(predicted)).sum()),
    }


def load_model(path: Path, compute_unit: str) -> tuple[ct.models.MLModel, float]:
    unit = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
    }[compute_unit]
    started = time.perf_counter_ns()
    model = ct.models.MLModel(str(path), compute_units=unit)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    return model, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--init-model", default="WhamApp/WhamApp/WHAM_I.mlpackage", type=Path)
    parser.add_argument("--keypoint-model", default="WhamApp/WhamApp/WHAM_S.mlpackage", type=Path)
    parser.add_argument("--image-model", default="WhamApp/WhamApp/WHAM_ImageStep.mlpackage", type=Path)
    parser.add_argument("--pose-init-model", type=Path)
    parser.add_argument("--sequences", default=3, type=int)
    parser.add_argument("--frames", default=120, type=int)
    parser.add_argument("--compute-unit", choices=("all", "cpu", "cpu-gpu", "cpu-ne"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = joblib.load(args.dataset)
    count = min(args.sequences, len(dataset["vid"]))
    sequences = [prepare_sequence(dataset, index, args.frames) for index in range(count)]

    init_model, init_load_ms = load_model(args.init_model, args.compute_unit)
    keypoint_model, keypoint_load_ms = load_model(args.keypoint_model, args.compute_unit)
    image_model, image_load_ms = load_model(args.image_model, args.compute_unit)
    pose_init_model = None
    pose_init_load_ms = None
    if args.pose_init_model:
        pose_init_model, pose_init_load_ms = load_model(args.pose_init_model, args.compute_unit)

    variants = {
        "former_app_zero_init_keypoints_only": dict(
            step_model=keypoint_model,
            image_model=False,
            initialization="legacy_zero",
            image_valid=False,
        ),
        "keypoints_only_with_official_init": dict(
            step_model=keypoint_model,
            image_model=False,
            initialization="official",
            image_valid=False,
        ),
        "image_step_feature_ablation": dict(
            step_model=image_model,
            image_model=True,
            initialization="official",
            image_valid=False,
        ),
        "image_step_official_features": dict(
            step_model=image_model,
            image_model=True,
            initialization="official",
            image_valid=True,
        ),
        "image_step_mobile_neutral_init": dict(
            step_model=image_model,
            image_model=True,
            initialization="mobile_neutral",
            image_valid=True,
        ),
    }
    if pose_init_model is not None:
        variants["image_step_zero_3d_with_official_hmr2_pose"] = dict(
            step_model=image_model,
            image_model=True,
            initialization="mobile_feature_pose",
            image_valid=True,
        )

    results: dict[str, Any] = {}
    stored_predictions: dict[str, list[np.ndarray]] = {}
    for name, configuration in variants.items():
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        latency: list[float] = []
        for sequence in sequences:
            predicted, times = run_sequence(
                init_model,
                sequence=sequence,
                pose_init_model=pose_init_model,
                **configuration,
            )
            predictions.append(predicted)
            targets.append(sequence.target_pose_axis_angle)
            latency.extend(times)
        stored_predictions[name] = predictions
        results[name] = summarize_variant(predictions, targets, latency)

    feature_delta = np.concatenate(stored_predictions["image_step_official_features"], axis=0) - np.concatenate(
        stored_predictions["image_step_feature_ablation"], axis=0
    )
    results["image_connection_check"] = {
        "pose_output_mean_abs_delta": float(np.abs(feature_delta).mean()),
        "pose_output_max_abs_delta": float(np.abs(feature_delta).max()),
        "passed": bool(np.abs(feature_delta).max() > 1e-5),
    }
    if pose_init_model is not None:
        predicted_initial_pose = np.concatenate(
            [
                pose_init_model.predict({"image_feature": sequence.initial_feature})[
                    "smpl_pose_6d"
                ]
                for sequence in sequences
            ],
            axis=0,
        )
        target_initial_pose = np.concatenate(
            [sequence.init_pose_axis_angle for sequence in sequences], axis=0
        )
        initial_error = geodesic_degrees(predicted_initial_pose, target_initial_pose)
        results["pose_initializer_on_official_hmr2_features"] = {
            "sequences": len(sequences),
            "pose_rotation_error_deg": float(initial_error.mean()),
            "body_rotation_error_deg": float(initial_error[:, 1:].mean()),
            "root_rotation_error_deg": float(initial_error[:, 0].mean()),
            "note": "Validates the exported HMR2 readout using official HMR2 features; it does not measure the distilled FastViT feature error.",
        }
    report = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "dataset_sequences": [sequence.name for sequence in sequences],
        "frames_per_sequence_cap": args.frames,
        "metric_note": "Rotation errors are an asset-free diagnostic, not the paper's SMPL MPJPE/PVE metrics.",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "compute_unit": args.compute_unit,
        },
        "model_load_latency_ms": {
            "WHAM_I": init_load_ms,
            "WHAM_S_keypoints_only": keypoint_load_ms,
            "WHAM_ImageStep": image_load_ms,
            **({"FastViTInitialPose": pose_init_load_ms} if pose_init_load_ms is not None else {}),
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
