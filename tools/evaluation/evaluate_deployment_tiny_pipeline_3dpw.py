#!/usr/bin/env python3
"""Locked 3DPW test for the trained compact iPhone pipeline.

This evaluator loads the validation-selected deployment checkpoint and runs the
single intended path: YOLO26 -> FastViT -> learned compact initializer -> WHAM.
No HMR2 model/readout and no SMPL initialization decode are used.  Licensed
SMPL assets are loaded only after prediction to calculate the paper metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_full_pipeline_tradeoff as full_eval
import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as feature_eval
import joblib
import numpy as np
import torch
from train_deployment_tiny_pipeline import (
    DEPLOYMENT_SCHEMA,
    SOURCE_STUDENT_SHA256,
    YOLO26_SHA256,
    DeploymentInitializer,
    deployment_core,
)
from ultralytics import YOLO

VARIANT = "tiny_yolo26_fastvit_learned_init_adapted_wham"

# Exact single-person population measured by the saved released-WHAM run.
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
REFERENCE_REPORT_SHA256 = (
    "ce0205077f4487281c04c10ae5f9a8e963e2c76d482491b23c71f902a5f75f4f"
)
REFERENCE_METRICS = {
    "pa_mpjpe_mm": 32.66937072048713,
    "mpjpe_mm": 55.57580911372661,
    "pve_mm": 65.6364644886124,
    "accel_official_30fps": 6.226210307692702,
}


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


def load_deployment_models(
    checkpoint_path: Path,
    wham_repo: Path,
    wham_checkpoint: Path,
    device: torch.device,
) -> tuple[
    feature_eval.FastViTHMR2Student,
    DeploymentInitializer,
    torch.nn.Module,
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("deployment_schema") != DEPLOYMENT_SCHEMA:
        raise RuntimeError(
            f"Expected deployment schema {DEPLOYMENT_SCHEMA}, found "
            f"{checkpoint.get('deployment_schema')}"
        )
    metadata = checkpoint.get("deployment_metadata") or {}
    if metadata.get("source_checkpoint_sha256") != SOURCE_STUDENT_SHA256:
        raise RuntimeError(
            "Deployment checkpoint did not originate from the pinned phase-three model"
        )
    if metadata.get("test_data_used") is not False:
        raise RuntimeError(
            "Training provenance does not prove that 3DPW test was untouched"
        )
    student, _ = feature_eval.load_student(checkpoint_path, device)
    config = checkpoint.get("deployment_initializer_config") or {}
    initializer = DeploymentInitializer(hidden_dim=int(config.get("hidden_dim", 512)))
    initializer.load_state_dict(
        checkpoint["deployment_initializer_state_dict"], strict=True
    )
    initializer.eval().to(device)
    network = feature_eval.load_wham_core(wham_repo, wham_checkpoint, device)
    network.load_state_dict(checkpoint["deployment_wham_state_dict"], strict=True)
    network.eval()
    return student.eval(), initializer, network, checkpoint


@torch.inference_mode()
def learned_initialization(
    initializer: DeploymentInitializer,
    first_token: torch.Tensor,
    first_keypoints: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    pose, joints = initializer(first_token.reshape(1, 1, 1024).to(device))
    init_kp = torch.cat(
        (joints.reshape(1, 1, 51), first_keypoints.reshape(1, 1, 37).to(device)),
        dim=-1,
    )
    return (
        {
            "init_kp": init_kp,
            "init_pose": pose,
            "init_root": pose[:, :, 0],
        },
        {
            "pose_6d_l2": float(torch.linalg.vector_norm(pose).cpu()),
            "joints_3d_l2": float(torch.linalg.vector_norm(joints).cpu()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--parsed-3dpw", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--deployment-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo26-weights", type=Path, required=True)
    parser.add_argument("--smpl-model-directory", type=Path, required=True)
    parser.add_argument("--h36m-joint-regressor", type=Path, required=True)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-output", type=Path)
    args = parser.parse_args()

    required = (
        args.parsed_3dpw,
        args.deployment_checkpoint,
        args.wham_checkpoint,
        args.yolo26_weights,
        args.h36m_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required inputs: " + ", ".join(missing))
    yolo_sha = feature_eval.sha256_file(args.yolo26_weights)
    if yolo_sha != YOLO26_SHA256:
        raise RuntimeError(f"YOLO26 checksum mismatch: {yolo_sha}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if commit != feature_eval.WHAM_COMMIT:
        raise RuntimeError(f"Unexpected WHAM commit: {commit}")
    if str(args.wham_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.wham_repo.resolve()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; locked test variant: {VARIANT}", flush=True)
    image_root = feature_eval.locate_image_root(args.three_dpw_root.resolve())
    labels = joblib.load(args.parsed_3dpw)
    selected = selected_indices(labels)
    student, initializer, network, deployment_checkpoint = load_deployment_models(
        args.deployment_checkpoint,
        args.wham_repo.resolve(),
        args.wham_checkpoint,
        device,
    )
    pose_model = YOLO(str(args.yolo26_weights))
    smpl_models = full_eval.load_smpl_models(
        args.smpl_model_directory.resolve(), device
    )
    h36m_np = np.load(args.h36m_joint_regressor)[full_eval.H36M_TO_J14, :]
    h36m_regressor = torch.from_numpy(h36m_np).float().unsqueeze(0).to(device)

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
        initialization, diagnostics = learned_initialization(
            initializer, features[0], x[0], device
        )
        full_eval.synchronize(device)
        work_seconds["fastvit_initializer_heads"] += time.perf_counter() - init_started
        initializer_diagnostics.append(diagnostics)
        inputs = {
            "x": x[1:].unsqueeze(0).to(device),
            "mask": mask[1:].unsqueeze(0).to(device),
            "features": features[1:].unsqueeze(0).to(device),
            **initialization,
            "cam_angvel": torch.zeros(1, available, 6, device=device),
        }
        feature_valid = (features[1:] != 0).any(dim=-1).unsqueeze(0).to(device)
        output, core_seconds = full_eval.timed(
            device,
            lambda current_inputs=inputs, current_valid=feature_valid: deployment_core(
                network=network,
                feature_valid=current_valid,
                **current_inputs,
            ),
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
                    "pa_mpjpe_mm": row["pa_mpjpe_mm"],
                    "mpjpe_mm": row["mpjpe_mm"],
                }
            ),
            flush=True,
        )

    metrics = {name: summarize(values) for name, values in accumulated.items()}
    expected_recurrent_frames = sum(REFERENCE_TRACKS.values())
    if metrics["mpjpe_mm"]["samples"] != expected_recurrent_frames:
        raise RuntimeError("Tiny/reference sample population mismatch")
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
                "YOLO26n-pose + deployment-tuned FastViT + learned compact pose/3D-joint "
                "initializer + validation-selected input-adapted split WHAM"
            ),
            "hmr2_used_anywhere_in_tiny_inference": False,
            "smpl_use": "metric calculation only; not part of tiny inference",
            "camera_motion": "zero because 3DPW does not contain the phone gyroscope stream",
            "test_run_policy": "one locked run after validation selection; do not tune from this result",
        },
        "detection": {
            "detections": total_detections,
            "source_frames": total_source_frames,
            "rate": total_detections / max(total_source_frames, 1),
        },
        "initializer_diagnostics": {
            key: feature_eval.summarize(
                np.asarray(
                    [row[key] for row in initializer_diagnostics], dtype=np.float32
                )
            )
            for key in initializer_diagnostics[0]
        },
        "work_seconds": dict(work_seconds),
        "training_selection": {
            "epoch": deployment_checkpoint.get("deployment_epoch"),
            "stage": deployment_checkpoint.get("deployment_stage"),
            "validation": deployment_checkpoint.get("deployment_validation"),
        },
        "provenance": {
            "wham_commit": commit,
            "wham_checkpoint_sha256": feature_eval.sha256_file(args.wham_checkpoint),
            "deployment_checkpoint_sha256": feature_eval.sha256_file(
                args.deployment_checkpoint
            ),
            "source_phase3_checkpoint_sha256": SOURCE_STUDENT_SHA256,
            "yolo26_weights_sha256": yolo_sha,
            "parsed_3dpw_sha256": feature_eval.sha256_file(args.parsed_3dpw),
            "h36m_joint_regressor_sha256": feature_eval.sha256_file(
                args.h36m_joint_regressor
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
