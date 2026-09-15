#!/usr/bin/env python3
"""Evaluate only the complete tiny iPhone pipeline on the saved 3DPW subset.

The released-WHAM reference was already measured, so this evaluator does not
run it again.  It runs the final compact path:

    YOLO26n-pose -> phase-three FastViT -> FastViT/HMR2 linear initialization
    -> WHAM initializer -> causal WHAM image step

HMR2's 2.7 GB backbone is never executed.  Its small frozen pose and shape
linear readouts are applied to the FastViT token.  Licensed SMPL is evaluated
once per sequence to turn that first-frame pose/shape into WHAM's 17 initial
3D joints, and after inference to calculate the paper's metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as functional
from ultralytics import YOLO

import evaluate_full_pipeline_tradeoff as full_eval
import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as feature_eval


VARIANT = "tiny_yolo26_fastvit_complete_init"
HMR2_SHA256 = "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"
YOLO26_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
STUDENT_SHA256 = "f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1"

# These are exactly the single-person tracks scored in the already completed
# released-WHAM run. Pinning names and lengths prevents an accidental change of
# population from masquerading as an accuracy change.
REFERENCE_TRACKS = {
    "downtown_walkBridge_01_0": 1178,
    "flat_guitar_01_0": 747,
    "downtown_walkUphill_00_0": 384,
    "downtown_upstairs_00_0": 824,
    "downtown_downstairs_00_0": 627,
    "downtown_weeklyMarket_00_0": 1042,
    "outdoors_fencing_01_0": 924,
    "flat_packBags_00_0": 1272,
    "downtown_stairs_00_0": 1189,
    "downtown_windowShopping_00_0": 1805,
    "downtown_enterShop_00_0": 1357,
}
REFERENCE_REPORT_SHA256 = "ce0205077f4487281c04c10ae5f9a8e963e2c76d482491b23c71f902a5f75f4f"
REFERENCE_METRICS = {
    "pa_mpjpe_mm": 32.66937072048713,
    "mpjpe_mm": 55.57580911372661,
    "pve_mm": 65.6364644886124,
    "accel_official_30fps": 6.226210307692702,
}


class FastViTInitializerReadout:
    """The compact HMR2 linear heads consumed by the FastViT token."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        state = checkpoint["state_dict"]
        self.pose_weight = state["smpl_head.decpose.weight"].float().to(device)
        self.pose_bias = state["smpl_head.decpose.bias"].float().to(device)
        self.mean_pose = state["smpl_head.init_body_pose"].float().to(device)
        self.shape_weight = state["smpl_head.decshape.weight"].float().to(device)
        self.shape_bias = state["smpl_head.decshape.bias"].float().to(device)
        self.mean_shape = state["smpl_head.init_betas"].float().to(device)
        del checkpoint, state

    def __call__(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = token.reshape(-1, 1024).float()
        pose = functional.linear(token, self.pose_weight, self.pose_bias) + self.mean_pose
        shape = (
            functional.linear(token, self.shape_weight, self.shape_bias)
            + self.mean_shape
        )
        return pose.reshape(-1, 24, 6), shape.reshape(-1, 10)


def hmr2_rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Match HMR2's column-vector 6D conversion exactly."""

    pair = rotation.reshape(-1, 2, 3).permute(0, 2, 1)
    first = functional.normalize(pair[:, :, 0], dim=-1)
    second_raw = pair[:, :, 1] - (
        (first * pair[:, :, 1]).sum(dim=-1, keepdim=True) * first
    )
    second = functional.normalize(second_raw, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1).reshape(-1, 24, 3, 3)


@torch.inference_mode()
def complete_fastvit_initialization(
    first_token: torch.Tensor,
    first_keypoints: torch.Tensor,
    readout: FastViTInitializerReadout,
    neutral_smpl: torch.nn.Module,
    wham_joint_regressor: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    raw_pose, betas = readout(first_token.to(device))
    matrices = hmr2_rotation_6d_to_matrix(raw_pose)
    smpl_output = neutral_smpl(
        global_orient=matrices[:, :1],
        body_pose=matrices[:, 1:],
        betas=betas,
        pose2rot=False,
    )
    joints = torch.matmul(wham_joint_regressor, smpl_output.vertices)[..., :17, :]
    pelvis = joints[..., [12, 11], :].mean(dim=-2, keepdim=True)
    centered_joints = joints - pelvis
    wham_pose = feature_eval.matrix_to_rotation_6d(matrices).reshape(1, 1, 24, 6)
    init_kp = torch.cat(
        (centered_joints.reshape(1, 1, 51), first_keypoints.reshape(1, 1, 37).to(device)),
        dim=-1,
    )
    return (
        {
            "init_kp": init_kp,
            "init_pose": wham_pose,
            "init_root": wham_pose[:, :, 0, :],
        },
        {
            "pose_6d_l2": float(torch.linalg.vector_norm(raw_pose).cpu()),
            "betas_l2": float(torch.linalg.vector_norm(betas).cpu()),
            "joints_3d_l2": float(torch.linalg.vector_norm(centered_joints).cpu()),
        },
    )


def summarize(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("Metric accumulator is empty")
    return feature_eval.summarize(np.concatenate(values))


def selected_indices(labels: dict[str, Any]) -> list[int]:
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, raw_name in enumerate(labels["vid"]):
        by_name[str(raw_name)].append(index)
    missing = [name for name in REFERENCE_TRACKS if len(by_name[name]) != 1]
    if missing:
        raise RuntimeError(f"Could not uniquely resolve reference tracks: {missing}")
    return [by_name[name][0] for name in REFERENCE_TRACKS]


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--parsed-3dpw", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr2-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo26-weights", type=Path, required=True)
    parser.add_argument("--smpl-model-directory", type=Path, required=True)
    parser.add_argument("--h36m-joint-regressor", type=Path, required=True)
    parser.add_argument("--wham-joint-regressor", type=Path, required=True)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-output", type=Path)
    args = parser.parse_args()

    required_files = (
        args.parsed_3dpw,
        args.student_checkpoint,
        args.hmr2_checkpoint,
        args.wham_checkpoint,
        args.yolo26_weights,
        args.h36m_joint_regressor,
        args.wham_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))

    hashes = {
        "student_checkpoint": feature_eval.sha256_file(args.student_checkpoint),
        "hmr2_checkpoint": feature_eval.sha256_file(args.hmr2_checkpoint),
        "yolo26_weights": feature_eval.sha256_file(args.yolo26_weights),
    }
    expected_hashes = {
        "student_checkpoint": STUDENT_SHA256,
        "hmr2_checkpoint": HMR2_SHA256,
        "yolo26_weights": YOLO26_SHA256,
    }
    mismatched = {
        name: {"expected": expected_hashes[name], "actual": actual}
        for name, actual in hashes.items()
        if actual != expected_hashes[name]
    }
    if mismatched:
        raise RuntimeError(f"Pinned model checksum mismatch: {mismatched}")

    commit = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if commit != feature_eval.WHAM_COMMIT:
        raise RuntimeError(
            f"Expected WHAM commit {feature_eval.WHAM_COMMIT}, found {commit}"
        )
    if str(args.wham_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.wham_repo.resolve()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; evaluating only {VARIANT}", flush=True)
    image_root = feature_eval.locate_image_root(args.three_dpw_root.resolve())
    labels = joblib.load(args.parsed_3dpw)
    selected = selected_indices(labels)

    student, student_training = feature_eval.load_student(
        args.student_checkpoint, device
    )
    network = feature_eval.load_wham_core(
        args.wham_repo.resolve(), args.wham_checkpoint, device
    )
    pose_model = YOLO(str(args.yolo26_weights))
    smpl_models = full_eval.load_smpl_models(
        args.smpl_model_directory.resolve(), device
    )
    readout = FastViTInitializerReadout(args.hmr2_checkpoint, device)
    h36m_np = np.load(args.h36m_joint_regressor)[full_eval.H36M_TO_J14, :]
    h36m_regressor = torch.from_numpy(h36m_np).float().unsqueeze(0).to(device)
    wham_regressor = (
        torch.from_numpy(np.load(args.wham_joint_regressor))
        .float()
        .unsqueeze(0)
        .to(device)
    )

    accumulated: dict[str, list[np.ndarray]] = defaultdict(list)
    per_sequence: list[dict[str, Any]] = []
    initializer_diagnostics: list[dict[str, float]] = []
    total_source_frames = 0
    total_detections = 0
    work_seconds: defaultdict[str, float] = defaultdict(float)

    for index in selected:
        video_id = str(labels["vid"][index])
        expected_frames = REFERENCE_TRACKS[video_id]
        available = full_eval.available_frames(labels, index)
        if available != expected_frames:
            raise RuntimeError(
                f"{video_id} has {available} recurrent frames; expected {expected_frames}"
            )
        frame_ids = feature_eval.to_numpy(labels["frame_id"][index])
        paths = feature_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: available + 1]
        )
        x, mask, features, observation_stats = mobile_eval.mobile_observations(
            pose_model,
            student,
            paths,
            args.pose_batch_size,
            args.student_batch_size,
            device,
        )

        init_started = time.perf_counter()
        initialization, init_diagnostic = complete_fastvit_initialization(
            features[0],
            x[0],
            readout,
            smpl_models["neutral"],
            wham_regressor,
            device,
        )
        full_eval.synchronize(device)
        work_seconds["fastvit_initializer_readout_and_smpl"] += (
            time.perf_counter() - init_started
        )
        initializer_diagnostics.append(init_diagnostic)

        inputs = {
            "x": x[1:].unsqueeze(0).to(device),
            "mask": mask[1:].unsqueeze(0).to(device),
            "features": features[1:].unsqueeze(0).to(device),
            **initialization,
            # Match the original saved 3DPW evaluation convention. 3DPW does
            # not provide the phone's recorded gyroscope stream.
            "cam_angvel": torch.zeros(1, available, 6, device=device),
        }
        output, core_seconds = full_eval.timed(
            device, lambda: full_eval.run_core(network, inputs)
        )
        work_seconds["yolo26_pose"] += observation_stats["pose_frontend_seconds"]
        work_seconds["fastvit"] += observation_stats["fastvit_seconds"]
        work_seconds["wham_core"] += core_seconds

        target_pose = torch.from_numpy(
            feature_eval.to_numpy(labels["pose"][index])[1 : available + 1].astype(
                np.float32
            )
        )
        target_betas = torch.from_numpy(
            feature_eval.to_numpy(labels["betas"][index])[1 : available + 1].astype(
                np.float32
            )
        )
        gender = str(labels["gender"][index]).lower()
        metrics, decode_seconds = full_eval.smpl_metrics(
            output,
            target_pose,
            target_betas,
            gender,
            smpl_models,
            h36m_regressor,
            device,
            args.smpl_batch_size,
        )
        work_seconds["smpl_evaluation_decode"] += decode_seconds
        row: dict[str, Any] = {
            "variant": VARIANT,
            "sequence": video_id,
            "frames": available,
            "detections": observation_stats["detections"],
            "detection_rate": observation_stats["detections"] / len(paths),
        }
        for metric_name, values in metrics.items():
            accumulated[metric_name].append(values)
            row[metric_name] = float(values.mean()) if len(values) else None
        per_sequence.append(row)
        total_source_frames += len(paths)
        total_detections += int(observation_stats["detections"])
        print(
            json.dumps(
                {
                    "sequence": video_id,
                    "frames": available,
                    "detections": observation_stats["detections"],
                    "mpjpe_mm": row["mpjpe_mm"],
                    "pa_mpjpe_mm": row["pa_mpjpe_mm"],
                }
            ),
            flush=True,
        )

    metrics = {name: summarize(values) for name, values in accumulated.items()}
    expected_recurrent_frames = sum(REFERENCE_TRACKS.values())
    if metrics["mpjpe_mm"]["samples"] != expected_recurrent_frames:
        raise RuntimeError(
            "Tiny/reference sample mismatch: "
            f"{metrics['mpjpe_mm']['samples']} vs {expected_recurrent_frames}"
        )
    differences = {
        name: metrics[name]["mean"] - REFERENCE_METRICS[name]
        for name in REFERENCE_METRICS
    }
    relative_changes = {
        name: metrics[name]["mean"] / REFERENCE_METRICS[name] - 1.0
        for name in REFERENCE_METRICS
    }
    report = {
        "schema_version": 1,
        "comparison": {
            "reference": "saved released-WHAM flip result on the exact same 11 tracks",
            "reference_report_sha256": REFERENCE_REPORT_SHA256,
            "reference_metrics": REFERENCE_METRICS,
            "tiny_variant": VARIANT,
            "tiny_metrics": metrics,
            "tiny_minus_original": differences,
            "tiny_relative_change_vs_original": relative_changes,
        },
        "scope": {
            "dataset": "registered 3DPW test",
            "tracks": len(REFERENCE_TRACKS),
            "track_frames": REFERENCE_TRACKS,
            "recurrent_frames": expected_recurrent_frames,
            "source_frames_including_initializer": total_source_frames,
            "pipeline": (
                "YOLO26n-pose + phase-three FastViT per-frame features + first-frame "
                "FastViT token decoded by the frozen HMR2 pose/shape linear heads + "
                "one SMPL initialization decode + split WHAM_I/WHAM_ImageStep"
            ),
            "no_hmr2_backbone_inference": True,
            "camera_motion": (
                "zero for both sides under the saved official 3DPW evaluation convention; "
                "physical iPhone run will use recorded device gyro"
            ),
        },
        "detection": {
            "detections": total_detections,
            "source_frames": total_source_frames,
            "rate": total_detections / max(total_source_frames, 1),
        },
        "initializer_diagnostics": {
            key: feature_eval.summarize(
                np.asarray([row[key] for row in initializer_diagnostics], dtype=np.float32)
            )
            for key in initializer_diagnostics[0]
        },
        "work_seconds": dict(work_seconds),
        "provenance": {
            "wham_commit": commit,
            "wham_checkpoint_sha256": feature_eval.sha256_file(args.wham_checkpoint),
            "student_checkpoint_sha256": hashes["student_checkpoint"],
            "student_training": student_training,
            "hmr2_checkpoint_sha256": hashes["hmr2_checkpoint"],
            "hmr2_checkpoint_use": "readout tensor extraction only; backbone not instantiated",
            "yolo26_weights_sha256": hashes["yolo26_weights"],
            "parsed_3dpw_sha256": feature_eval.sha256_file(args.parsed_3dpw),
            "h36m_joint_regressor_sha256": feature_eval.sha256_file(
                args.h36m_joint_regressor
            ),
            "wham_joint_regressor_sha256": feature_eval.sha256_file(
                args.wham_joint_regressor
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.per_sequence_output or args.output.with_suffix(".csv")
    fieldnames = [
        "variant",
        "sequence",
        "frames",
        "detections",
        "detection_rate",
        "pa_mpjpe_mm",
        "mpjpe_mm",
        "pve_mm",
        "accel_official_30fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_sequence)
    print(json.dumps(report["comparison"], indent=2), flush=True)
    print(f"Report: {args.output}\nPer-sequence: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
