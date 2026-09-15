#!/usr/bin/env python3
"""Frozen, apples-to-apples 3DPW evaluation of WHAM and the phone candidate.

There are exactly two rows:

1. released WHAM using its stored ViTPose/HMR2a inputs and flip average;
2. YOLO26n-pose + released HMR2.0-S + the same released WHAM weights.

Nothing is trained, fitted, selected, or calibrated.  Both rows are evaluated
on the identical single-person 3DPW tracks and recurrent frames.  SMPL is used
only after inference to calculate the paper's camera-relative body metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_mobile_pipeline_3dpw as mobile_preprocess
import evaluate_wham_feature_substitution as wham_eval
import joblib
import numpy as np
import torch
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
from tqdm import tqdm
from ultralytics import YOLO

VARIANTS = ("released_wham", "frozen_phone_candidate")
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def concatenate_summary(parts: list[np.ndarray]) -> dict[str, float | int]:
    if not parts:
        raise RuntimeError("No values were accumulated for a metric")
    return wham_eval.summarize(np.concatenate(parts))


class FrozenSMPLInitializer(nn.Module):
    """Neutral HMR2.0-S SMPL -> WHAM's root-centred first 17 joints."""

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
        matrices = wham_eval.rotation_6d_to_matrix(
            pose_6d.reshape(-1, 24, 6)
        )
        vertices, _ = lbs(
            betas.reshape(-1, 10),
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
        return joints - pelvis


def model_device_argument(device: torch.device) -> int | str:
    return 0 if device.type == "cuda" else "cpu"


@torch.inference_mode()
def frozen_phone_observations(
    pose_model: YOLO,
    hmr2s: FrozenHMR2S,
    image_paths: list[Path],
    pose_batch_size: int,
    hmr_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Apply the same detector threshold, square crop, and normalization as iOS."""

    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    crops: list[torch.Tensor | None] = []
    confidences: list[float] = []
    pose_seconds = 0.0

    for start in tqdm(
        range(0, len(image_paths), pose_batch_size),
        desc="YOLO26 observations",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        originals: list[Image.Image] = []
        detector_images: list[Image.Image] = []
        for path in image_paths[start : start + pose_batch_size]:
            with Image.open(path) as source:
                original = source.convert("RGB")
                originals.append(original.copy())
                detector_images.append(
                    original.resize((640, 640), Image.Resampling.BILINEAR)
                )

        started = time.perf_counter()
        results = pose_model.predict(
            detector_images,
            imgsz=640,
            conf=0.001,
            device=model_device_argument(device),
            half=device.type == "cuda",
            verbose=False,
        )
        synchronize(device)
        pose_seconds += time.perf_counter() - started

        for original, result in zip(originals, results):
            width, height = original.size
            best_index = -1
            confidence = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                confidence_values = result.boxes.conf.detach().float().cpu().numpy()
                best_index = int(np.argmax(confidence_values))
                confidence = float(confidence_values[best_index])

            if confidence <= 0.5 or result.keypoints is None or best_index < 0:
                x_rows.append(np.zeros(37, dtype=np.float32))
                mask_rows.append(np.ones(17, dtype=np.float32))
                crops.append(None)
                confidences.append(confidence)
                continue

            keypoint_data = (
                result.keypoints.data[best_index].detach().float().cpu().numpy()
            )
            keypoints = keypoint_data[:, :2] * np.asarray(
                [width / 640.0, height / 640.0], dtype=np.float32
            )
            keypoint_confidence = keypoint_data[:, 2]
            valid = keypoint_confidence >= 0.3
            box_xyxy = (
                result.boxes.xyxy[best_index].detach().float().cpu().numpy()
                * np.asarray(
                    [width / 640.0, height / 640.0, width / 640.0, height / 640.0],
                    dtype=np.float32,
                )
            )
            if int(valid.sum()) >= 7:
                minimum = keypoints[valid].min(axis=0)
                maximum = keypoints[valid].max(axis=0)
                center_x, center_y = ((minimum + maximum) / 2.0).tolist()
                side = max(float(np.max(maximum - minimum)) * 1.2, 1.0)
            else:
                center_x = float((box_xyxy[0] + box_xyxy[2]) / 2.0)
                center_y = float((box_xyxy[1] + box_xyxy[3]) / 2.0)
                side = max(
                    float(
                        max(
                            box_xyxy[2] - box_xyxy[0],
                            box_xyxy[3] - box_xyxy[1],
                        )
                    )
                    * 1.05,
                    1.0,
                )

            coordinates = 2.0 * (
                keypoints - np.asarray([center_x, center_y], dtype=np.float32)
            ) / side
            longest = float(max(width, height))
            location = np.asarray(
                [
                    2.0 * center_x / longest - width / longest,
                    2.0 * center_y / longest - height / longest,
                    side / longest,
                ],
                dtype=np.float32,
            )
            x_rows.append(
                np.concatenate((coordinates.reshape(34), location)).astype(np.float32)
            )
            mask_rows.append((keypoint_confidence < 0.3).astype(np.float32))
            crop = mobile_preprocess.square_crop(
                original, center_x, center_y, side, size=256
            )
            crops.append(mobile_preprocess.image_to_tensor(crop))
            confidences.append(confidence)

    count = len(crops)
    tokens = torch.zeros(count, 1024, dtype=torch.float32)
    poses = torch.zeros(count, 24, 6, dtype=torch.float32)
    betas = torch.zeros(count, 10, dtype=torch.float32)
    cameras = torch.zeros(count, 3, dtype=torch.float32)
    valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
    hmr_seconds = 0.0
    for start in tqdm(
        range(0, len(valid_indices), hmr_batch_size),
        desc="HMR2.0-S",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        indices = valid_indices[start : start + hmr_batch_size]
        batch = torch.stack([crops[index] for index in indices]).to(
            device, non_blocking=True
        )
        synchronize(device)
        started = time.perf_counter()
        batch_token, batch_pose, batch_betas, batch_camera = hmr2s(batch)
        synchronize(device)
        hmr_seconds += time.perf_counter() - started
        tokens[indices] = batch_token.float().cpu()
        poses[indices] = batch_pose.float().cpu()
        betas[indices] = batch_betas.float().cpu()
        cameras[indices] = batch_camera.float().cpu()

    confidence_array = np.asarray(confidences, dtype=np.float32)
    valid_array = confidence_array > 0.5
    return {
        "x": torch.from_numpy(np.stack(x_rows)),
        "mask": torch.from_numpy(np.stack(mask_rows)).bool(),
        "token": tokens,
        "pose": poses,
        "betas": betas,
        "camera": cameras,
        "valid": torch.from_numpy(valid_array.astype(np.float32)).reshape(-1, 1),
        "frames": count,
        "detections": int(valid_array.sum()),
        "detection_confidence": wham_eval.summarize(confidence_array),
        "seconds": {"yolo26": pose_seconds, "hmr2s": hmr_seconds},
    }


@torch.inference_mode()
def run_phone_core(
    network: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
    feature_valid: torch.Tensor,
    init_kp: torch.Tensor,
    init_pose: torch.Tensor,
    init_root: torch.Tensor,
    cam_angvel: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the frozen WHAM body path with the exported miss-detector behavior."""

    processed = network.preprocess(x.clone(), mask)
    pred_kp3d, motion_context = network.motion_encoder(processed, init_kp)
    pred_root, pred_vel = network.trajectory_decoder(
        motion_context, init_root, cam_angvel
    )
    integrator_input = torch.cat((motion_context, features), dim=-1)
    integrated = network.integrator.layer1(integrator_input)
    integrated = network.integrator.relu1(integrated)
    integrated = network.integrator.layer2(integrated)
    integrated = network.integrator.relu2(integrated)
    integrated = network.integrator.layer3(integrated)
    image_integrated = integrated + motion_context
    valid = feature_valid.clamp(0.0, 1.0)
    fused = valid * image_integrated + (1.0 - valid) * motion_context
    pred_pose, pred_shape, pred_cam, pred_contact = network.motion_decoder(
        fused, init_pose
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


def available_frames(labels: dict[str, Any], index: int) -> int:
    return max(
        min(
            len(labels[key][index]) - 1
            for key in (
                "kp2d",
                "bbox",
                "features",
                "flipped_kp2d",
                "flipped_bbox",
                "flipped_features",
                "pose",
                "betas",
                "frame_id",
            )
        ),
        0,
    )


def summarize_variants(
    accumulated: dict[str, dict[str, list[np.ndarray]]],
    timings: dict[str, dict[str, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        result[variant] = {
            "metrics": {
                name: concatenate_summary(parts)
                for name, parts in accumulated[variant].items()
            },
            "work_seconds": timings[variant],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--parsed-3dpw", required=True, type=Path)
    parser.add_argument("--three-dpw-root", required=True, type=Path)
    parser.add_argument("--wham-repo", required=True, type=Path)
    parser.add_argument("--wham-checkpoint", required=True, type=Path)
    parser.add_argument("--hmr2s-repo", required=True, type=Path)
    parser.add_argument("--hmr2s-checkpoint", required=True, type=Path)
    parser.add_argument("--yolo26-weights", required=True, type=Path)
    parser.add_argument("--smpl-model-directory", required=True, type=Path)
    parser.add_argument("--h36m-joint-regressor", required=True, type=Path)
    parser.add_argument("--wham-joint-regressor", required=True, type=Path)
    parser.add_argument("--sequences", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--hmr-batch-size", type=int, default=32)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-sequence-output", type=Path)
    args = parser.parse_args()

    required = (
        args.parsed_3dpw,
        args.wham_checkpoint,
        args.hmr2s_checkpoint,
        args.yolo26_weights,
        args.h36m_joint_regressor,
        args.wham_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
        args.hmr2s_repo / "4D-Humans/hmr2/models/backbones/vit.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))
    if sha256(args.hmr2s_checkpoint) != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError("The HMR2.0-S checkpoint is not the pinned official release")
    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    hmr2s_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.hmr2s_repo, text=True
    ).strip()
    if wham_commit != wham_eval.WHAM_COMMIT:
        raise RuntimeError(
            f"Expected WHAM {wham_eval.WHAM_COMMIT}, found {wham_commit}"
        )
    if hmr2s_commit != HMR2S_COMMIT:
        raise RuntimeError(f"Expected HMR2.0-S {HMR2S_COMMIT}, found {hmr2s_commit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = wham_eval.locate_image_root(args.three_dpw_root.resolve())
    print(f"Device: {device}; raw images: {image_root}", flush=True)
    labels = joblib.load(args.parsed_3dpw)
    required_labels = {
        "vid",
        "frame_id",
        "res",
        "kp2d",
        "bbox",
        "features",
        "flipped_kp2d",
        "flipped_bbox",
        "flipped_features",
        "init_kp3d",
        "init_pose",
        "flipped_init_kp3d",
        "flipped_init_pose",
        "pose",
        "betas",
        "gender",
    }
    absent = sorted(required_labels - set(labels))
    if absent:
        raise RuntimeError(f"Parsed 3DPW file is missing fields: {absent}")

    matching = [
        index
        for index, raw_video in enumerate(labels["vid"])
        if (image_root / str(raw_video).rsplit("_", 1)[0]).is_dir()
    ]
    counts = Counter(str(labels["vid"][i]).rsplit("_", 1)[0] for i in matching)
    single_person = [
        index
        for index in matching
        if counts[str(labels["vid"][index]).rsplit("_", 1)[0]] == 1
    ]
    if args.sequences > 0 and len(single_person) > args.sequences:
        positions = np.linspace(
            0, len(single_person) - 1, args.sequences, dtype=np.int64
        )
        selected = [single_person[int(position)] for position in positions]
    else:
        selected = single_person
    if not selected:
        raise RuntimeError("No single-person 3DPW track matched the raw image dataset")
    print(
        f"Protocol population: {len(selected)} single-person tracks; no training or selection",
        flush=True,
    )

    network = wham_eval.load_wham_core(
        args.wham_repo.resolve(), args.wham_checkpoint, device
    )
    hmr2s = FrozenHMR2S(
        args.hmr2s_repo.resolve(), args.hmr2s_checkpoint
    ).to(device).eval()
    pose_model = YOLO(str(args.yolo26_weights))
    initializer = FrozenSMPLInitializer(
        checkpoint_smpl_buffers(args.hmr2s_checkpoint),
        torch.from_numpy(np.load(args.wham_joint_regressor)).float(),
    ).to(device).eval()
    smpl_models = reference_eval.load_smpl_models(
        args.smpl_model_directory.resolve(), device
    )
    h36m = torch.from_numpy(
        np.load(args.h36m_joint_regressor)[H36M_TO_J14]
    ).float().unsqueeze(0).to(device)

    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        variant: defaultdict(list) for variant in VARIANTS
    }
    timings: dict[str, dict[str, float]] = {
        variant: defaultdict(float) for variant in VARIANTS
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    detections = 0
    source_frames = 0

    for index in tqdm(
        selected,
        desc="3DPW single-person tracks",
        disable=not sys.stderr.isatty(),
    ):
        video_id = str(labels["vid"][index])
        frames = available_frames(labels, index)
        if args.frames > 0:
            frames = min(frames, args.frames)
        if frames <= 0:
            skipped.append({"sequence": video_id, "reason": "no recurrent frames"})
            continue
        frame_ids = wham_eval.to_numpy(labels["frame_id"][index])
        paths = wham_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: frames + 1]
        )
        observations = frozen_phone_observations(
            pose_model,
            hmr2s,
            paths,
            args.pose_batch_size,
            args.hmr_batch_size,
            device,
        )
        source_frames += int(observations["frames"])
        detections += int(observations["detections"])
        if observations["valid"][0, 0].item() < 0.5:
            skipped.append(
                {"sequence": video_id, "reason": "no person on initializer frame"}
            )
            print(f"Skipping {video_id}: no first-frame detection", flush=True)
            continue

        first_pose = observations["pose"][:1].to(device)
        first_betas = observations["betas"][:1].to(device)
        init_kp3d = initializer(first_pose, first_betas).reshape(1, 1, 51)
        init_kp = torch.cat(
            (init_kp3d, observations["x"][:1].to(device).reshape(1, 1, 37)),
            dim=-1,
        )
        phone_inputs = {
            "x": observations["x"][1:].unsqueeze(0).to(device),
            "mask": observations["mask"][1:].unsqueeze(0).to(device),
            "features": observations["token"][1:].unsqueeze(0).to(device),
            "feature_valid": observations["valid"][1:].unsqueeze(0).to(device),
            "init_kp": init_kp,
            "init_pose": first_pose.reshape(1, 1, 24, 6),
            "init_root": first_pose[:, 0, :].reshape(1, 1, 6),
            "cam_angvel": torch.zeros(1, frames, 6, device=device),
        }
        synchronize(device)
        started = time.perf_counter()
        phone_output = run_phone_core(network, **phone_inputs)
        synchronize(device)
        timings["frozen_phone_candidate"]["wham_body_core"] += (
            time.perf_counter() - started
        )
        for name, seconds in observations["seconds"].items():
            timings["frozen_phone_candidate"][name] += seconds

        normal_inputs = reference_eval.official_inputs(
            labels, index, frames, "", device
        )
        flipped_inputs = reference_eval.official_inputs(
            labels, index, frames, "flipped_", device
        )
        synchronize(device)
        started = time.perf_counter()
        normal_output = reference_eval.run_core(network, normal_inputs)
        flipped_output = reference_eval.run_core(network, flipped_inputs)
        released_output = reference_eval.average_flipped_prediction(
            args.wham_repo.resolve(), normal_output, flipped_output
        )
        synchronize(device)
        timings["released_wham"]["wham_body_core_and_flip"] += (
            time.perf_counter() - started
        )

        target_pose = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        target_betas = torch.from_numpy(
            wham_eval.to_numpy(labels["betas"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        gender = str(labels["gender"][index]).lower()
        sequence_summary: dict[str, Any] = {}
        for variant, prediction in (
            ("released_wham", released_output),
            ("frozen_phone_candidate", phone_output),
        ):
            metrics, decode_seconds = reference_eval.smpl_metrics(
                prediction,
                target_pose,
                target_betas,
                gender,
                smpl_models,
                h36m,
                device,
                args.smpl_batch_size,
            )
            timings[variant]["smpl_metric_decode"] += decode_seconds
            row: dict[str, Any] = {
                "variant": variant,
                "sequence": video_id,
                "frames": frames,
            }
            if variant == "frozen_phone_candidate":
                row["detections"] = observations["detections"]
                row["detection_rate"] = observations["detections"] / len(paths)
            for metric, values in metrics.items():
                accumulated[variant][metric].append(values)
                row[metric] = float(values.mean()) if len(values) else None
            rows.append(row)
            sequence_summary[variant] = row
        print(
            json.dumps(
                {
                    "sequence": video_id,
                    "frames": frames,
                    "released_pa_mpjpe_mm": sequence_summary["released_wham"][
                        "pa_mpjpe_mm"
                    ],
                    "phone_pa_mpjpe_mm": sequence_summary[
                        "frozen_phone_candidate"
                    ]["pa_mpjpe_mm"],
                    "detection_rate": observations["detections"] / len(paths),
                }
            ),
            flush=True,
        )

    if not rows:
        raise RuntimeError("No 3DPW track completed both frozen pipelines")
    variants = summarize_variants(accumulated, timings)
    released = variants["released_wham"]["metrics"]
    phone = variants["frozen_phone_candidate"]["metrics"]
    compared_metrics = (
        "pa_mpjpe_mm",
        "mpjpe_mm",
        "pve_mm",
        "accel_official_30fps",
    )
    absolute_change = {
        metric: phone[metric]["mean"] - released[metric]["mean"]
        for metric in compared_metrics
    }
    relative_change = {
        metric: phone[metric]["mean"] / max(released[metric]["mean"], 1e-9) - 1.0
        for metric in compared_metrics
    }
    completed_sequences = len(
        {row["sequence"] for row in rows if row["variant"] == "released_wham"}
    )
    report = {
        "schema_version": 1,
        "experiment": "frozen_released_wham_vs_frozen_phone_candidate",
        "training_performed": False,
        "protocol": {
            "dataset": "3DPW test",
            "population": "all matching single-person tracks with a valid initializer-frame YOLO26 detection",
            "selected_tracks_before_detection_guard": len(selected),
            "completed_same_population_tracks": completed_sequences,
            "recurrent_frames_per_variant": released["mpjpe_mm"]["samples"],
            "maximum_frames_per_track": args.frames,
            "released_wham": "stored official ViTPose/HMR2a observations and initialization; flip average; released WHAM weights",
            "frozen_phone_candidate": "YOLO26n-pose; released HMR2.0-S 1024-D token and first-frame SMPL initialization; released WHAM weights; no flip",
            "same_population": True,
            "world_grounded": False,
            "camera_signal": "zero angular velocity, matching WHAM's 3DPW protocol; prerecorded 3DPW has no phone gyroscope stream",
            "metric_decode": "gendered licensed SMPL and H36M J14 regressor, evaluation only",
            "latency_warning": "work_seconds is batched Kaggle throughput and is not iPhone latency",
            "refiner_note": "WHAM's contact trajectory refiner changes world trajectory, not the camera-relative pose/shape used by these four metrics",
        },
        "provenance": {
            "wham_repository": "https://github.com/yohanshin/WHAM.git",
            "wham_commit": wham_commit,
            "wham_checkpoint_sha256": sha256(args.wham_checkpoint),
            "hmr2s_repository": HMR2S_REPOSITORY,
            "hmr2s_commit": hmr2s_commit,
            "hmr2s_checkpoint_sha256": sha256(args.hmr2s_checkpoint),
            "hmr2s_parameter_count": sum(
                parameter.numel() for parameter in hmr2s.parameters()
            ),
            "yolo26_weights_sha256": sha256(args.yolo26_weights),
            "parsed_3dpw_sha256": sha256(args.parsed_3dpw),
            "h36m_regressor_sha256": sha256(args.h36m_joint_regressor),
            "wham_joint_regressor_sha256": sha256(args.wham_joint_regressor),
        },
        "detection": {
            "detections": detections,
            "source_frames_including_skipped_tracks": source_frames,
            "rate": detections / max(source_frames, 1),
            "initializer_track_success_rate": completed_sequences
            / max(len(selected), 1),
            "skipped_tracks": skipped,
        },
        "variants": variants,
        "phone_minus_released_same_population": absolute_change,
        "phone_relative_change_vs_released_same_population": relative_change,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.per_sequence_output or args.output.with_suffix(".csv")
    fields = [
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
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "training_performed": False,
                "completed_tracks": completed_sequences,
                "released": {
                    metric: released[metric]["mean"] for metric in compared_metrics
                },
                "phone": {
                    metric: phone[metric]["mean"] for metric in compared_metrics
                },
                "phone_minus_released": absolute_change,
                "report": str(args.output),
                "per_sequence": str(csv_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
