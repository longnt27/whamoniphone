#!/usr/bin/env python3
"""Measure the 3DPW accuracy tradeoff from released WHAM to the iPhone subset.

The evaluated rows deliberately separate two effects:

1. the released WHAM path with its official flip evaluation;
2. the same released path without flip evaluation;
3. only the HMR2 image feature replaced by the distilled FastViT token; and
4. the proposed iPhone subset (YOLOv8 pose, FastViT, neutral initialization).

The first three rows use every person track. The app-like row and a separately
reported WHAM reference use every single-person video, because the current app
does not associate identities in multi-person video. All rows are decoded with
licensed SMPL assets *for evaluation only* and scored with the same PA-MPJPE,
MPJPE, PVE, and acceleration code used by WHAM's 3DPW evaluator. The iPhone
application does not contain this SMPL evaluation decode. 3DPW's official WHAM
evaluator uses zero camera angular velocity, so none of the rows below is a
world-grounded trajectory evaluation.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as feature_eval


VARIANTS = (
    "paper_wham_bedlam_flip",
    "paper_wham_bedlam_no_flip",
    "fastvit_official_inputs",
    "iphone_subset_yolov8",
)
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
PELVIS_INDICES = [2, 3]
FPS = 30


def patch_legacy_smpl_dependencies() -> None:
    """Allow licensed SMPL v1 pickles to load on modern Python/NumPy."""

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    try:
        __import__("chumpy")
    except ImportError:
        # SMPL .npz files do not need chumpy.  A clearer error will be raised by
        # smplx if a supplied legacy .pkl actually needs it.
        pass


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(device: torch.device, operation: Any) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, time.perf_counter() - started


def concatenate_summary(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("Metric accumulator is empty")
    return feature_eval.summarize(np.concatenate(values))


def available_frames(labels: dict[str, Any], index: int) -> int:
    frame_lengths = [
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
    ]
    return max(min(frame_lengths), 0)


def official_inputs(
    labels: dict[str, Any],
    index: int,
    frames: int,
    prefix: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keypoints = feature_eval.to_numpy(labels[prefix + "kp2d"][index])
    bbox = feature_eval.to_numpy(labels[prefix + "bbox"][index])
    resolution = feature_eval.to_numpy(labels["res"][index][0])
    x_all, mask_all = feature_eval.normalized_wham_input(
        keypoints, bbox, resolution
    )
    frame_slice = slice(1, frames + 1)

    init_kp3d = feature_eval.root_center_coco(
        feature_eval.to_numpy(labels[prefix + "init_kp3d"][index])[0]
    )
    init_kp = np.concatenate((init_kp3d.reshape(-1), x_all[0]), axis=0)
    init_pose_axis_angle = torch.from_numpy(
        feature_eval.to_numpy(labels[prefix + "init_pose"][index])[0].astype(
            np.float32
        )
    ).reshape(24, 3)
    init_pose = feature_eval.matrix_to_rotation_6d(
        feature_eval.axis_angle_to_matrix(init_pose_axis_angle)
    ).reshape(1, 1, 24, 6)

    target_first_root = torch.from_numpy(
        feature_eval.to_numpy(labels["pose"][index])[0].astype(np.float32)
    ).reshape(24, 3)[0]
    init_root = feature_eval.matrix_to_rotation_6d(
        feature_eval.axis_angle_to_matrix(target_first_root)
    ).reshape(1, 1, 6)
    return {
        "x": torch.from_numpy(x_all[frame_slice]).unsqueeze(0).to(device),
        "mask": torch.from_numpy(mask_all[frame_slice]).unsqueeze(0).to(device),
        "features": torch.from_numpy(
            feature_eval.to_numpy(labels[prefix + "features"][index])[
                frame_slice
            ].astype(np.float32)
        )
        .unsqueeze(0)
        .to(device),
        "init_kp": torch.from_numpy(init_kp.astype(np.float32))
        .reshape(1, 1, 88)
        .to(device),
        "init_pose": init_pose.to(device),
        "init_root": init_root.to(device),
        "cam_angvel": torch.zeros(1, frames, 6, device=device),
    }


def run_core(
    network: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    features: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return feature_eval.run_wham_core(
        network=network,
        x=inputs["x"],
        mask=inputs["mask"],
        features=inputs["features"] if features is None else features,
        init_kp=inputs["init_kp"],
        init_pose=inputs["init_pose"],
        init_root=inputs["init_root"],
        cam_angvel=inputs["cam_angvel"],
    )


def average_flipped_prediction(
    wham_repo: Path,
    normal: dict[str, torch.Tensor],
    flipped: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if str(wham_repo) not in sys.path:
        sys.path.insert(0, str(wham_repo))
    from lib.utils.imutils import avg_preds  # pylint: disable=import-outside-toplevel

    normal_pose = normal["pose"].squeeze(0).reshape(-1, 24, 6)
    flipped_pose = flipped["pose"].squeeze(0).reshape(-1, 24, 6)
    average_pose, average_shape = avg_preds(
        normal_pose,
        normal["shape"].squeeze(0),
        flipped_pose,
        flipped["shape"].squeeze(0),
    )
    return {
        "pose": average_pose.reshape_as(normal["pose"]),
        "shape": average_shape.reshape_as(normal["shape"]),
    }


def load_smpl_models(
    model_directory: Path, device: torch.device
) -> dict[str, torch.nn.Module]:
    patch_legacy_smpl_dependencies()
    from smplx import SMPL  # pylint: disable=import-outside-toplevel

    if not all(
        (model_directory / f"SMPL_{gender.upper()}.pkl").is_file()
        for gender in ("male", "female", "neutral")
    ):
        raise RuntimeError(
            "SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, and SMPL_FEMALE.pkl are "
            f"required in {model_directory}"
        )
    models: dict[str, torch.nn.Module] = {}
    for gender in ("male", "female", "neutral"):
        model_file = model_directory / f"SMPL_{gender.upper()}.pkl"
        models[gender] = SMPL(
            model_path=str(model_file),
            gender=gender,
            create_transl=False,
        ).eval().to(device)
    return models


@torch.inference_mode()
def smpl_metrics(
    prediction: dict[str, torch.Tensor],
    target_pose_axis_angle: torch.Tensor,
    target_betas: torch.Tensor,
    gender: str,
    models: dict[str, torch.nn.Module],
    joint_regressor: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], float]:
    predicted_pose = feature_eval.rotation_6d_to_matrix(
        prediction["pose"].reshape(-1, 24, 6).float()
    ).to(device)
    predicted_betas = prediction["shape"].reshape(-1, 10).float().to(device)
    target_pose = feature_eval.axis_angle_to_matrix(
        target_pose_axis_angle.reshape(-1, 24, 3).float()
    ).to(device)
    target_betas = target_betas.reshape(-1, 10).float().to(device)
    if len(predicted_pose) != len(target_pose):
        raise RuntimeError(
            f"Prediction/target length mismatch: {len(predicted_pose)} and "
            f"{len(target_pose)}"
        )

    pa_parts: list[np.ndarray] = []
    mpjpe_parts: list[np.ndarray] = []
    pve_parts: list[np.ndarray] = []
    predicted_joint_parts: list[torch.Tensor] = []
    target_joint_parts: list[torch.Tensor] = []
    started = time.perf_counter()
    neutral_model = models["neutral"]
    target_model = models[gender]
    for start in range(0, len(predicted_pose), chunk_size):
        stop = min(start + chunk_size, len(predicted_pose))
        pred_output = neutral_model(
            body_pose=predicted_pose[start:stop, 1:],
            global_orient=predicted_pose[start:stop, :1],
            betas=predicted_betas[start:stop],
            pose2rot=False,
        )
        target_output = target_model(
            body_pose=target_pose[start:stop, 1:],
            global_orient=target_pose[start:stop, :1],
            betas=target_betas[start:stop],
            pose2rot=False,
        )
        predicted_vertices = pred_output.vertices
        target_vertices = target_output.vertices
        predicted_joints = torch.matmul(
            joint_regressor, predicted_vertices
        )
        target_joints = torch.matmul(joint_regressor, target_vertices)

        predicted_pelvis = predicted_joints[:, PELVIS_INDICES].mean(
            dim=1, keepdim=True
        )
        target_pelvis = target_joints[:, PELVIS_INDICES].mean(
            dim=1, keepdim=True
        )
        predicted_joints = predicted_joints - predicted_pelvis
        target_joints = target_joints - target_pelvis
        predicted_vertices = predicted_vertices - predicted_pelvis
        target_vertices = target_vertices - target_pelvis

        from lib.eval.eval_utils import (  # pylint: disable=import-outside-toplevel
            batch_compute_similarity_transform_torch,
        )

        aligned = batch_compute_similarity_transform_torch(
            predicted_joints, target_joints
        )
        pa_parts.append(
            (
                torch.linalg.vector_norm(aligned - target_joints, dim=-1)
                .mean(dim=-1)
                .cpu()
                .numpy()
                * 1000.0
            )
        )
        mpjpe_parts.append(
            (
                torch.linalg.vector_norm(
                    predicted_joints - target_joints, dim=-1
                )
                .mean(dim=-1)
                .cpu()
                .numpy()
                * 1000.0
            )
        )
        pve_parts.append(
            (
                torch.linalg.vector_norm(
                    predicted_vertices - target_vertices, dim=-1
                )
                .mean(dim=-1)
                .cpu()
                .numpy()
                * 1000.0
            )
        )
        predicted_joint_parts.append(predicted_joints.cpu())
        target_joint_parts.append(target_joints.cpu())
        del pred_output, target_output, predicted_vertices, target_vertices

    synchronize(device)
    elapsed = time.perf_counter() - started
    predicted_joints = torch.cat(predicted_joint_parts).numpy()
    target_joints = torch.cat(target_joint_parts).numpy()
    if len(predicted_joints) >= 5:
        acceleration = (
            np.linalg.norm(
                (predicted_joints[:-2] - 2 * predicted_joints[1:-1] + predicted_joints[2:])
                - (target_joints[:-2] - 2 * target_joints[1:-1] + target_joints[2:]),
                axis=2,
            ).mean(axis=1)[1:-1]
            * (FPS**2)
        )
    else:
        acceleration = np.empty(0, dtype=np.float32)
    return (
        {
            "pa_mpjpe_mm": np.concatenate(pa_parts),
            "mpjpe_mm": np.concatenate(mpjpe_parts),
            "pve_mm": np.concatenate(pve_parts),
            "accel_official_30fps": acceleration,
        },
        elapsed,
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
    parser.add_argument("--smpl-model-directory", type=Path, required=True)
    parser.add_argument("--h36m-joint-regressor", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-output", type=Path)
    args = parser.parse_args()

    required_files = (
        args.parsed_3dpw,
        args.student_checkpoint,
        args.wham_checkpoint,
        args.yolov8_weights,
        args.h36m_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))
    for name in ("SMPL_NEUTRAL", "SMPL_MALE", "SMPL_FEMALE"):
        if not (args.smpl_model_directory / f"{name}.pkl").is_file():
            parser.error(
                f"Missing {name}.pkl in {args.smpl_model_directory}"
            )

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
    print(f"Device: {device}", flush=True)
    image_root = feature_eval.locate_image_root(args.three_dpw_root.resolve())
    labels = joblib.load(args.parsed_3dpw)
    required_label_keys = {
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
        "vid",
        "frame_id",
        "res",
    }
    absent = sorted(required_label_keys - set(labels))
    if absent:
        raise RuntimeError(f"Parsed 3DPW is missing official fields: {absent}")

    matching = [
        index
        for index, raw_video_id in enumerate(labels["vid"])
        if (image_root / str(raw_video_id).rsplit("_", 1)[0]).is_dir()
    ]
    if not matching:
        raise RuntimeError("No parsed 3DPW tracks matched the raw image directory")
    if args.sequences > 0:
        count = min(args.sequences, len(matching))
        positions = np.linspace(0, len(matching) - 1, count, dtype=np.int64)
        selected = [matching[int(position)] for position in positions]
    else:
        selected = matching
    track_counts_by_video = Counter(
        str(labels["vid"][index]).rsplit("_", 1)[0] for index in matching
    )
    mobile_eligible = {
        index
        for index in selected
        if track_counts_by_video[str(labels["vid"][index]).rsplit("_", 1)[0]]
        == 1
    }
    if not mobile_eligible:
        raise RuntimeError(
            "The selected tracks contain no single-person 3DPW videos. The current "
            "iPhone subset selects only one highest-confidence person, so it cannot "
            "be scored honestly against multiple annotated identities."
        )

    student, student_training = feature_eval.load_student(
        args.student_checkpoint, device
    )
    network = feature_eval.load_wham_core(
        args.wham_repo.resolve(), args.wham_checkpoint, device
    )
    process_image = feature_eval.load_process_image(args.wham_repo.resolve())
    pose_model = YOLO(str(args.yolov8_weights))
    models = load_smpl_models(args.smpl_model_directory.resolve(), device)
    regressor_np = np.load(args.h36m_joint_regressor)[H36M_TO_J14, :]
    joint_regressor = torch.from_numpy(regressor_np).float().unsqueeze(0).to(device)

    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        variant: defaultdict(list) for variant in VARIANTS
    }
    timings: dict[str, defaultdict[str, float]] = {
        variant: defaultdict(float) for variant in VARIANTS
    }
    per_sequence: list[dict[str, Any]] = []
    controlled_drift: list[np.ndarray] = []
    mobile_reference_accumulated: dict[str, list[np.ndarray]] = defaultdict(list)
    total_mobile_source_frames = 0
    total_mobile_detections = 0

    for index in tqdm(selected, desc="3DPW tracks"):
        video_id = str(labels["vid"][index])
        frames = available_frames(labels, index)
        if args.frames > 0:
            frames = min(frames, args.frames)
        if frames <= 0:
            continue
        frame_ids = feature_eval.to_numpy(labels["frame_id"][index])
        paths = feature_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: frames + 1]
        )
        normal_inputs = official_inputs(labels, index, frames, "", device)
        flipped_inputs = official_inputs(labels, index, frames, "flipped_", device)

        normal_output, normal_seconds = timed(
            device, lambda: run_core(network, normal_inputs)
        )
        flipped_output, flipped_seconds = timed(
            device, lambda: run_core(network, flipped_inputs)
        )
        average_started = time.perf_counter()
        flip_output = average_flipped_prediction(
            args.wham_repo.resolve(), normal_output, flipped_output
        )
        average_seconds = time.perf_counter() - average_started

        student_features, feature_seconds = feature_eval.extract_student_features(
            student,
            paths[1:],
            feature_eval.to_numpy(labels["bbox"][index])[1 : frames + 1],
            process_image,
            args.student_batch_size,
            device,
        )
        student_output, student_core_seconds = timed(
            device,
            lambda: run_core(
                network,
                normal_inputs,
                student_features.unsqueeze(0).to(device),
            ),
        )

        predictions = {
            "paper_wham_bedlam_flip": flip_output,
            "paper_wham_bedlam_no_flip": normal_output,
            "fastvit_official_inputs": student_output,
        }
        timings["paper_wham_bedlam_flip"]["wham_core"] += (
            normal_seconds + flipped_seconds + average_seconds
        )
        timings["paper_wham_bedlam_no_flip"]["wham_core"] += normal_seconds
        timings["fastvit_official_inputs"]["fastvit"] += feature_seconds
        timings["fastvit_official_inputs"]["wham_core"] += student_core_seconds
        observation_stats: dict[str, Any] | None = None
        if index in mobile_eligible:
            mobile_x, mobile_mask, mobile_features, observation_stats = (
                mobile_eval.mobile_observations(
                    pose_model,
                    student,
                    paths,
                    args.pose_batch_size,
                    args.student_batch_size,
                    device,
                )
            )
            mobile_initial_keypoints = torch.cat(
                (torch.zeros(51, dtype=torch.float32), mobile_x[0]), dim=0
            ).reshape(1, 1, 88)
            mobile_inputs = {
                "x": mobile_x[1:].unsqueeze(0).to(device),
                "mask": mobile_mask[1:].unsqueeze(0).to(device),
                "features": mobile_features[1:].unsqueeze(0).to(device),
                "init_kp": mobile_initial_keypoints.to(device),
                "init_pose": mobile_eval.neutral_pose(device),
                "init_root": mobile_eval.identity_root(device),
                "cam_angvel": torch.zeros(1, frames, 6, device=device),
            }
            mobile_output, mobile_core_seconds = timed(
                device, lambda: run_core(network, mobile_inputs)
            )
            predictions["iphone_subset_yolov8"] = mobile_output
            timings["iphone_subset_yolov8"]["yolo_pose"] += observation_stats[
                "pose_frontend_seconds"
            ]
            timings["iphone_subset_yolov8"]["fastvit"] += observation_stats[
                "fastvit_seconds"
            ]
            timings["iphone_subset_yolov8"]["wham_core"] += mobile_core_seconds

        target_pose = torch.from_numpy(
            feature_eval.to_numpy(labels["pose"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        target_betas = torch.from_numpy(
            feature_eval.to_numpy(labels["betas"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        gender = str(labels["gender"][index]).lower()
        if gender not in models:
            raise RuntimeError(f"Unsupported 3DPW gender {gender!r} for {video_id}")

        sequence_rows: dict[str, dict[str, Any]] = {}
        for variant, prediction in predictions.items():
            metrics, smpl_seconds = smpl_metrics(
                prediction,
                target_pose,
                target_betas,
                gender,
                models,
                joint_regressor,
                device,
                args.smpl_batch_size,
            )
            timings[variant]["smpl_evaluation_decode"] += smpl_seconds
            row: dict[str, Any] = {
                "variant": variant,
                "sequence": video_id,
                "frames": frames,
            }
            for metric_name, values in metrics.items():
                accumulated[variant][metric_name].append(values)
                row[metric_name] = float(values.mean()) if len(values) else None
            if variant == "iphone_subset_yolov8":
                assert observation_stats is not None
                row["detections"] = observation_stats["detections"]
                row["detection_rate"] = observation_stats["detections"] / len(paths)
            if variant == "paper_wham_bedlam_flip" and index in mobile_eligible:
                for metric_name, values in metrics.items():
                    mobile_reference_accumulated[metric_name].append(values)
            per_sequence.append(row)
            sequence_rows[variant] = row

        drift = feature_eval.pose_drift(
            student_output["pose"].float().cpu(),
            normal_output["pose"].float().cpu(),
        ).numpy()
        controlled_drift.append(drift)
        if observation_stats is not None:
            total_mobile_source_frames += len(paths)
            total_mobile_detections += int(observation_stats["detections"])
        print(
            json.dumps(
                {
                    "sequence": video_id,
                    "frames": frames,
                    "paper_pa_mpjpe_mm": sequence_rows[
                        "paper_wham_bedlam_flip"
                    ]["pa_mpjpe_mm"],
                    "iphone_subset_eligible": index in mobile_eligible,
                    "iphone_pa_mpjpe_mm": sequence_rows.get(
                        "iphone_subset_yolov8", {}
                    ).get("pa_mpjpe_mm"),
                    "iphone_detections": (
                        None
                        if observation_stats is None
                        else observation_stats["detections"]
                    ),
                }
            ),
            flush=True,
        )

    if not per_sequence:
        raise RuntimeError("No non-empty 3DPW tracks were evaluated")

    results: dict[str, Any] = {}
    for variant in VARIANTS:
        results[variant] = {
            "metrics": {
                metric: concatenate_summary(values)
                for metric, values in accumulated[variant].items()
            },
            "work_seconds": dict(timings[variant]),
        }
    paper = results["paper_wham_bedlam_flip"]["metrics"]
    no_flip = results["paper_wham_bedlam_no_flip"]["metrics"]
    controlled = results["fastvit_official_inputs"]["metrics"]
    iphone = results["iphone_subset_yolov8"]["metrics"]
    mobile_reference = {
        metric: concatenate_summary(values)
        for metric, values in mobile_reference_accumulated.items()
    }
    drift_summary = feature_eval.rotation_summary(np.concatenate(controlled_drift))
    paper_delta = {
        "pa_mpjpe_mm": paper["pa_mpjpe_mm"]["mean"] - 35.7,
        "mpjpe_mm": paper["mpjpe_mm"]["mean"] - 56.9,
        "pve_mm": paper["pve_mm"]["mean"] - 67.4,
        "accel_m_per_s2": (
            paper["accel_official_30fps"]["mean"] - 6.7
        ),
    }
    baseline_reproduced = (
        abs(paper_delta["pa_mpjpe_mm"]) <= 1.0
        and abs(paper_delta["mpjpe_mm"]) <= 1.0
        and abs(paper_delta["pve_mm"]) <= 1.0
        and abs(paper_delta["accel_m_per_s2"]) <= 0.3
    )
    report = {
        "schema_version": 1,
        "scope": {
            "dataset": "registered 3DPW test + official WHAM parsed ViT labels",
            "official_and_controlled_person_tracks": len(selected),
            "official_and_controlled_recurrent_frames": paper["mpjpe_mm"]["samples"],
            "iphone_single_person_tracks": len(mobile_eligible),
            "iphone_recurrent_frames": iphone["mpjpe_mm"]["samples"],
            "iphone_source_frames_including_initializer": total_mobile_source_frames,
            "population_note": (
                "Official WHAM and the controlled FastViT substitution use every "
                "selected person track. The app-like row and its explicit WHAM "
                "reference use all selected single-person videos because the current "
                "iPhone app selects one highest-confidence person and has no identity "
                "association for multi-person videos."
            ),
            "official_reference": (
                "released WHAM real+BEDLAM architecture/checkpoint, official stored ViTPose/HMR2 "
                "inputs, HMR2 initialization, ground-truth first root, and flip averaging"
            ),
            "iphone_subset": (
                "YOLOv8n-pose, phase-three FastViT, zero first-frame 3D joints, "
                "neutral SMPL pose, identity root, zero camera angular velocity"
            ),
            "smpl_evaluation_decode": (
                "Licensed SMPL is applied offline to every row only to calculate the "
                "paper metrics. It is not part of the current iPhone pipeline or latency."
            ),
            "world_grounded": False,
            "world_grounded_note": (
                "WHAM's 3DPW evaluator sets camera angular velocity to zero. Use EMDB "
                "split 2 for a future world-grounded trajectory comparison."
            ),
            "timing_note": (
                "work_seconds is batched Kaggle diagnostic throughput. Official HMR2 and "
                "ViTPose inputs are cached, so it is not an end-to-end latency comparison."
            ),
        },
        "paper_reference_for_this_bedlam_checkpoint": {
            "source": "WHAM CVPR 2024 Table 2, real datasets plus BEDLAM",
            "pa_mpjpe_mm": 35.7,
            "mpjpe_mm": 56.9,
            "pve_mm": 67.4,
            "accel_m_per_s2": 6.7,
        },
        "baseline_reproduction": {
            "passed": baseline_reproduced,
            "tolerance": {
                "position_metrics_mm": 1.0,
                "accel_m_per_s2": 0.3,
            },
            "measured_minus_paper": paper_delta,
        },
        "provenance": {
            "wham_commit": commit,
            "wham_checkpoint_sha256": feature_eval.sha256_file(
                args.wham_checkpoint
            ),
            "student_checkpoint_sha256": feature_eval.sha256_file(
                args.student_checkpoint
            ),
            "student_training": student_training,
            "parsed_3dpw_sha256": feature_eval.sha256_file(args.parsed_3dpw),
            "yolov8_weights_sha256": feature_eval.sha256_file(
                args.yolov8_weights
            ),
            "h36m_joint_regressor_sha256": feature_eval.sha256_file(
                args.h36m_joint_regressor
            ),
            "artifact_bytes": {
                "released_wham_checkpoint": args.wham_checkpoint.stat().st_size,
                "phase3_fastvit_training_checkpoint": (
                    args.student_checkpoint.stat().st_size
                ),
                "yolov8n_pose_pytorch_weights": args.yolov8_weights.stat().st_size,
            },
            "phase3_fastvit_parameters": sum(
                parameter.numel() for parameter in student.parameters()
            ),
        },
        "variants": results,
        "controlled_fastvit": {
            "student_teacher_pose_drift": drift_summary,
            "minus_paper_wham_bedlam_no_flip": {
                metric: controlled[metric]["mean"] - no_flip[metric]["mean"]
                for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
            },
        },
        "iphone_reference_paper_wham_single_person_subset": {
            "metrics": mobile_reference,
            "person_tracks": len(mobile_eligible),
        },
        "iphone_minus_paper_wham_same_single_person_subset": {
            metric: iphone[metric]["mean"] - mobile_reference[metric]["mean"]
            for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
        },
        "iphone_relative_change_vs_paper_wham_same_single_person_subset": {
            metric: (
                iphone[metric]["mean"]
                / max(mobile_reference[metric]["mean"], 1e-9)
                - 1.0
            )
            for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
        },
        "iphone_detection": {
            "detections": total_mobile_detections,
            "source_frames": total_mobile_source_frames,
            "rate": total_mobile_detections
            / max(total_mobile_source_frames, 1),
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
    print(
        json.dumps(
            {
                "paper_wham_bedlam_flip": {
                    metric: paper[metric]["mean"]
                    for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
                },
                "fastvit_official_inputs": {
                    metric: controlled[metric]["mean"]
                    for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
                },
                "iphone_subset_yolov8": {
                    metric: iphone[metric]["mean"]
                    for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")
                },
                "report": str(args.output),
                "per_sequence": str(csv_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
