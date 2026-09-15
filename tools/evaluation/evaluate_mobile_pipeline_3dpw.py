#!/usr/bin/env python3
"""Evaluate the app-like mobile WHAM subset on registered 3DPW images.

This complements evaluate_wham_feature_substitution.py. That evaluator isolates
the FastViT replacement with official keypoints and initialization; this one
uses a YOLO pose front end plus the same explicitly neutral initialization as
the iOS app. It reports rotation-space diagnostics and does not claim the
paper's SMPL MPJPE/PVE or world-grounded metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

import evaluate_wham_feature_substitution as wham_eval


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (torch.from_numpy(array) - IMAGENET_MEAN) / IMAGENET_STD


def square_crop(
    image: Image.Image,
    center_x: float,
    center_y: float,
    side: float,
    size: int = 256,
) -> Image.Image:
    scale = side / float(size)
    return image.transform(
        (size, size),
        Image.Transform.AFFINE,
        (
            scale,
            0.0,
            center_x - side / 2.0,
            0.0,
            scale,
            center_y - side / 2.0,
        ),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def model_device_argument(device: torch.device) -> int | str:
    return 0 if device.type == "cuda" else "cpu"


@torch.inference_mode()
def mobile_observations(
    pose_model: YOLO,
    student: wham_eval.FastViTHMR2Student,
    image_paths: list[Path],
    pose_batch_size: int,
    student_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    crops: list[torch.Tensor | None] = []
    confidences: list[float] = []
    pose_seconds = 0.0

    for start in tqdm(
        range(0, len(image_paths), pose_batch_size),
        desc="YOLO observations",
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        pose_seconds += time.perf_counter() - started

        for original, result in zip(originals, results):
            width, height = original.size
            if result.boxes is None or len(result.boxes) == 0:
                confidence = 0.0
            else:
                confidence_values = result.boxes.conf.detach().float().cpu().numpy()
                best_index = int(np.argmax(confidence_values))
                confidence = float(confidence_values[best_index])

            if confidence <= 0.5 or result.keypoints is None:
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
                    float(max(box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]))
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
            crops.append(image_to_tensor(square_crop(original, center_x, center_y, side)))
            confidences.append(confidence)

    features = torch.zeros(len(crops), 1024, dtype=torch.float32)
    valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
    feature_seconds = 0.0
    for start in range(0, len(valid_indices), student_batch_size):
        indices = valid_indices[start : start + student_batch_size]
        batch = torch.stack([crops[index] for index in indices if crops[index] is not None])
        batch = batch.to(device, non_blocking=True)
        started = time.perf_counter()
        output = student(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        feature_seconds += time.perf_counter() - started
        features[indices] = output.float().cpu()

    confidence_array = np.asarray(confidences, dtype=np.float32)
    return (
        torch.from_numpy(np.stack(x_rows)),
        torch.from_numpy(np.stack(mask_rows)).bool(),
        features,
        {
            "frames": len(image_paths),
            "detections": int((confidence_array > 0.5).sum()),
            "detection_confidence": wham_eval.summarize(confidence_array),
            "pose_frontend_seconds": pose_seconds,
            "fastvit_seconds": feature_seconds,
        },
    )


def neutral_pose(device: torch.device) -> torch.Tensor:
    identity = torch.eye(3, dtype=torch.float32, device=device)
    rotation = wham_eval.matrix_to_rotation_6d(identity).reshape(1, 1, 1, 6)
    return rotation.expand(1, 1, 24, 6).contiguous()


def identity_root(device: torch.device) -> torch.Tensor:
    return wham_eval.matrix_to_rotation_6d(
        torch.eye(3, dtype=torch.float32, device=device)
    ).reshape(1, 1, 6)


@torch.inference_mode()
def evaluate_variant(
    name: str,
    weights: Path,
    labels: dict[str, Any],
    selected: list[int],
    image_root: Path,
    student: wham_eval.FastViTHMR2Student,
    network: torch.nn.Module,
    pose_batch_size: int,
    student_batch_size: int,
    frame_limit: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pose_model = YOLO(str(weights))
    all_errors: list[torch.Tensor] = []
    per_sequence: list[dict[str, Any]] = []
    total_frames = 0
    total_detections = 0
    total_pose_seconds = 0.0
    total_feature_seconds = 0.0
    wham_seconds = 0.0

    for index in selected:
        video_id = str(labels["vid"][index])
        frame_ids = wham_eval.to_numpy(labels["frame_id"][index])
        available = min(
            len(frame_ids) - 1,
            len(labels["pose"][index]) - 1,
        )
        if frame_limit > 0:
            available = min(frame_limit, available)
        if available <= 0:
            continue
        paths = wham_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: available + 1]
        )
        x, mask, features, observation_stats = mobile_observations(
            pose_model,
            student,
            paths,
            pose_batch_size,
            student_batch_size,
            device,
        )
        initial_keypoints = torch.cat(
            (torch.zeros(51, dtype=torch.float32), x[0]), dim=0
        ).reshape(1, 1, 88)
        target = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1 : available + 1].astype(
                np.float32
            )
        ).reshape(available, 24, 3)

        started = time.perf_counter()
        output = wham_eval.run_wham_core(
            network=network,
            x=x[1:].unsqueeze(0).to(device),
            mask=mask[1:].unsqueeze(0).to(device),
            features=features[1:].unsqueeze(0).to(device),
            init_kp=initial_keypoints.to(device),
            init_pose=neutral_pose(device),
            init_root=identity_root(device),
            cam_angvel=torch.zeros(1, available, 6, device=device),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wham_seconds += time.perf_counter() - started
        errors = wham_eval.pose_error_against_axis_angle(
            output["pose"].float().cpu(), target
        )
        all_errors.append(errors)
        sequence_mean = float(errors.mean())
        per_sequence.append(
            {
                "variant": name,
                "sequence": video_id,
                "frames": available,
                "detections": observation_stats["detections"],
                "pose_error_deg": sequence_mean,
            }
        )
        total_frames += observation_stats["frames"]
        total_detections += observation_stats["detections"]
        total_pose_seconds += observation_stats["pose_frontend_seconds"]
        total_feature_seconds += observation_stats["fastvit_seconds"]

    if not all_errors:
        raise RuntimeError(f"No frames evaluated for {name}")
    errors = torch.cat(all_errors).numpy()
    weights_sha256 = wham_eval.sha256_file(weights)
    del pose_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        {
            "pose_model": weights.name,
            "pose_model_sha256": weights_sha256,
            "person_tracks": len(per_sequence),
            "source_frames_including_initializer": total_frames,
            "detections": total_detections,
            "detection_rate": total_detections / max(total_frames, 1),
            "pose_rotation_error": wham_eval.rotation_summary(errors),
            "gpu_work_seconds": {
                "pose_frontend": total_pose_seconds,
                "fastvit": total_feature_seconds,
                "wham": wham_seconds,
            },
        },
        per_sequence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--parsed-3dpw", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--yolov8-weights", type=Path, required=True)
    parser.add_argument("--yolo26-weights", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-output", type=Path)
    args = parser.parse_args()

    required = (
        args.parsed_3dpw,
        args.student_checkpoint,
        args.wham_checkpoint,
        args.yolov8_weights,
        args.yolo26_weights,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))

    wham_commit = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if wham_commit != wham_eval.WHAM_COMMIT:
        raise RuntimeError(
            f"Expected WHAM commit {wham_eval.WHAM_COMMIT}, found {wham_commit}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = wham_eval.locate_image_root(args.three_dpw_root.resolve())
    labels = joblib.load(args.parsed_3dpw)
    student, student_training = wham_eval.load_student(
        args.student_checkpoint, device
    )
    network = wham_eval.load_wham_core(
        args.wham_repo.resolve(), args.wham_checkpoint, device
    )
    matching = [
        index
        for index, raw_video_id in enumerate(labels["vid"])
        if (image_root / str(raw_video_id).rsplit("_", 1)[0]).is_dir()
    ]
    if not matching:
        raise RuntimeError("No parsed 3DPW tracks matched the raw image directory")
    if args.sequences <= 0:
        selected = matching
    else:
        count = min(args.sequences, len(matching))
        positions = np.linspace(0, len(matching) - 1, count, dtype=np.int64)
        selected = [matching[int(position)] for position in positions]

    variants: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, weights in (
        ("yolov8n_pose", args.yolov8_weights),
        ("yolo26n_pose", args.yolo26_weights),
    ):
        print(f"Evaluating {name} on {len(selected)} 3DPW tracks...", flush=True)
        result, variant_rows = evaluate_variant(
            name,
            weights,
            labels,
            selected,
            image_root,
            student,
            network,
            args.pose_batch_size,
            args.student_batch_size,
            args.frames,
            device,
        )
        variants[name] = result
        rows.extend(variant_rows)
        print(
            json.dumps(
                {
                    "variant": name,
                    "detection_rate": result["detection_rate"],
                    "pose_error_deg": result["pose_rotation_error"]["all_joints_deg"][
                        "mean"
                    ],
                },
                indent=2,
            ),
            flush=True,
        )

    v8_error = variants["yolov8n_pose"]["pose_rotation_error"]["all_joints_deg"][
        "mean"
    ]
    v26_error = variants["yolo26n_pose"]["pose_rotation_error"]["all_joints_deg"][
        "mean"
    ]
    report = {
        "schema_version": 1,
        "scope": {
            "dataset": "registered 3DPW test images + official WHAM parsed labels",
            "selected_tracks": len(selected),
            "frame_limit_per_track": args.frames,
            "initialization": "real first-frame YOLO 2D/crop + zero 3D joints + neutral SMPL pose + identity root",
            "camera": "zero angular velocity; no DPVO",
            "metric_note": "App-like root-relative rotation diagnostic; not PA-MPJPE/MPJPE/PVE or world-grounded WHAM.",
            "pose_runtime_note": "YOLO PyTorch weights are used for dataset accuracy; physical-iPhone Core ML latency is measured separately.",
        },
        "provenance": {
            "wham_commit": wham_commit,
            "wham_checkpoint_sha256": wham_eval.sha256_file(args.wham_checkpoint),
            "student_checkpoint_sha256": wham_eval.sha256_file(
                args.student_checkpoint
            ),
            "student_training": student_training,
            "parsed_3dpw_sha256": wham_eval.sha256_file(args.parsed_3dpw),
        },
        "variants": variants,
        "yolo26_minus_yolov8_pose_error_deg": v26_error - v8_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    polished = json.dumps(report, indent=2) + "\n"
    args.output.write_text(polished, encoding="utf-8")
    csv_path = args.per_sequence_output or args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "yolov8_pose_error_deg": v8_error,
                "yolo26_pose_error_deg": v26_error,
                "yolo26_minus_yolov8_pose_error_deg": v26_error - v8_error,
                "report": str(args.output),
                "per_sequence": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
