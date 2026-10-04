#!/usr/bin/env python3
"""Paired 3DPW test: locked phone pipeline with zero versus oracle camera rotation.

The 3DPW camera poses are injected only at WHAM's camera-angular-velocity input.
This is an oracle ablation, not a measured iPhone gyro/ARKit result.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_frozen_hmr2s_wham as frozen_eval
import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_hmr2s_temporal_smoothing as smoothing
import evaluate_wham_feature_substitution as wham_eval
import joblib
import numpy as np
import torch
from camera_motion_oracle import (
    camera_angular_velocity_with_validity,
    first_frame_aligned_world_joint_error,
    masked_first_valid_root_errors,
    rollout_root_translation,
)
from hmr2s_frozen import HMR2S_CHECKPOINT_SHA256, HMR2S_COMMIT, FrozenHMR2S, checkpoint_smpl_buffers
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

SCHEMA_VERSION = 2
VARIANTS = ("zero_camera_motion", "oracle_3dpw_camera_rotation")
ROOT_METRICS = ("root_orientation_error_deg", "root_displacement_error_mm")
WORLD_METRICS = (*ROOT_METRICS, "first_frame_aligned_world_mpjpe_mm")
POSE_METRICS = smoothing.METRICS
EXPECTED_TEST_TRACK_FRAMES = {
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
EXPECTED_LOCKED_LIGHT_BODY_METRICS = {
    "pa_mpjpe_mm": 51.7984296564426,
    "mpjpe_mm": 92.06258215296268,
    "pve_mm": 107.2021428112501,
    "accel_official_30fps": 10.722146670004268,
}
BODY_METRIC_REPRODUCTION_TOLERANCE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in (
        "test-parsed", "three-dpw-root", "wham-repo", "wham-checkpoint",
        "hmr2s-repo", "hmr2s-checkpoint", "yolo26m-weights", "adapter-checkpoint",
        "smpl-model-directory", "h36m-joint-regressor", "wham-joint-regressor",
        "output-json", "output-csv",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-yolo-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--pose-batch-size", type=int, default=24)
    parser.add_argument("--hmr-batch-size", type=int, default=24)
    parser.add_argument("--smpl-batch-size", type=int, default=192)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def locate_sequence_root(root: Path) -> Path:
    candidates = (
        root / "sequenceFiles" / "sequenceFiles" / "test",
        root / "sequenceFiles" / "test",
        root / "test",
    )
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one raw 3DPW test sequence directory; found {found}; checked {candidates}")
    return found[0]


def raw_track(sequence_root: Path, video_id: str) -> tuple[dict[str, Any], int]:
    sequence, person = video_id.rsplit("_", 1)
    path = sequence_root / f"{sequence}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw 3DPW sequence for {video_id}: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    person_index = int(person)
    for name in ("cam_poses", "campose_valid", "poses", "trans"):
        if name not in raw:
            raise KeyError(f"{path} lacks {name}")
    if person_index >= len(raw["poses"]) or person_index >= len(raw["trans"]):
        raise IndexError(f"Person {person_index} missing from {path}")
    return raw, person_index


def validate_raw_camera_track(
    parsed_camera_poses: np.ndarray,
    raw_camera_poses: np.ndarray,
    frame_ids: np.ndarray,
    camera_valid: np.ndarray,
) -> None:
    parsed = np.asarray(parsed_camera_poses, dtype=np.float32)
    raw = np.asarray(raw_camera_poses, dtype=np.float32)
    if parsed.ndim != 3 or parsed.shape[1:] != (4, 4):
        raise ValueError(f"Parsed camera poses have unexpected shape {parsed.shape}")
    if raw.ndim != 3 or raw.shape[1:] != (4, 4):
        raise ValueError(f"Raw camera poses have unexpected shape {raw.shape}")
    if len(parsed) != len(frame_ids) or frame_ids.min() < 0 or frame_ids.max() >= len(raw):
        raise ValueError("Parsed camera poses and raw frame IDs do not match")
    valid = np.asarray(camera_valid, dtype=bool).reshape(-1)
    if len(valid) != len(raw):
        raise ValueError("Raw camera-pose validity count mismatch")
    comparable = valid[frame_ids]
    if not np.allclose(parsed[comparable], raw[frame_ids[comparable]], rtol=1e-5, atol=1e-4):
        deviation = float(np.max(np.abs(parsed[comparable] - raw[frame_ids[comparable]])))
        raise ValueError(f"Parsed and raw camera poses disagree: maximum absolute difference {deviation}")


def summarize(parts: list[np.ndarray]) -> dict[str, float | int]:
    return frozen_eval.concatenate_summary(parts)


def root_errors(
    prediction: dict[str, torch.Tensor],
    initial_pose: torch.Tensor,
    target_world_pose: np.ndarray,
    target_world_translation: np.ndarray,
    world_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    predicted_root = wham_eval.rotation_6d_to_matrix(prediction["root"].reshape(-1, 6)).detach().cpu().numpy()
    initial_root = wham_eval.rotation_6d_to_matrix(initial_pose.reshape(-1, 6)).detach().cpu().numpy()[0]
    local_velocity = prediction["velocity"].reshape(-1, 3).detach().cpu().numpy()
    predicted_translation = rollout_root_translation(initial_root, predicted_root, local_velocity)
    target_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(predicted_root), axis=0)
    valid = np.asarray(world_valid, dtype=bool).reshape(-1)
    if valid.any():
        target_rotations[valid] = Rotation.from_rotvec(target_world_pose[valid, :3]).as_matrix()
    angular, displacement = masked_first_valid_root_errors(
        predicted_root, predicted_translation, target_rotations,
        target_world_translation, valid,
    )
    return {
        "root_orientation_error_deg": angular,
        "root_displacement_error_mm": displacement,
    }


def world_joint_error(
    filtered_prediction: dict[str, torch.Tensor],
    raw_prediction: dict[str, torch.Tensor],
    initial_pose: torch.Tensor,
    world_pose: np.ndarray,
    world_translation: np.ndarray,
    target_betas: torch.Tensor,
    gender: str,
    smpl_models: dict[str, torch.nn.Module],
    h36m_regressor: torch.Tensor,
    world_valid: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Decode full SMPL joints in world coordinates and score a rigid alignment."""

    frame_count = len(world_pose)
    pred_root = wham_eval.rotation_6d_to_matrix(
        raw_prediction["root"].reshape(-1, 6).float()
    ).to(device)
    init_root = wham_eval.rotation_6d_to_matrix(
        initial_pose.reshape(-1, 6).float()
    ).detach().cpu().numpy()[0]
    local_velocity = raw_prediction["velocity"].reshape(-1, 3).detach().cpu().numpy()
    predicted_translation = rollout_root_translation(
        init_root, pred_root.detach().cpu().numpy(), local_velocity
    )
    predicted_pose = wham_eval.rotation_6d_to_matrix(
        filtered_prediction["pose"].reshape(-1, 24, 6).float()
    ).to(device)
    if len(predicted_pose) != frame_count or len(pred_root) != frame_count:
        raise ValueError("World SMPL prediction and raw sequence lengths differ")
    safe_target_pose = np.asarray(world_pose[:, :72], dtype=np.float32).copy()
    safe_target_pose[~world_valid] = 0.0
    target_pose = wham_eval.axis_angle_to_matrix(
        torch.from_numpy(safe_target_pose).reshape(-1, 24, 3).to(device)
    )
    safe_target_translation = np.asarray(world_translation, dtype=np.float32).copy()
    safe_target_translation[~world_valid] = 0.0
    predicted_joints: list[np.ndarray] = []
    target_joints: list[np.ndarray] = []
    for start in range(0, frame_count, chunk_size):
        stop = min(start + chunk_size, frame_count)
        predicted = smpl_models["neutral"](
            body_pose=predicted_pose[start:stop, 1:],
            global_orient=pred_root[start:stop, None],
            betas=filtered_prediction["shape"].reshape(-1, 10)[start:stop].float().to(device),
            pose2rot=False,
        )
        target = smpl_models[gender](
            body_pose=target_pose[start:stop, 1:],
            global_orient=target_pose[start:stop, :1],
            betas=target_betas[start:stop].float().to(device),
            pose2rot=False,
        )
        predicted_world = torch.matmul(h36m_regressor, predicted.vertices) + torch.from_numpy(
            predicted_translation[start:stop]
        ).to(device)[:, None, :]
        target_world = torch.matmul(h36m_regressor, target.vertices) + torch.from_numpy(
            safe_target_translation[start:stop]
        ).to(device)[:, None, :]
        predicted_joints.append(predicted_world.detach().cpu().numpy())
        target_joints.append(target_world.detach().cpu().numpy())
    gt_root_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
    if world_valid.any():
        gt_root_rotations[world_valid] = Rotation.from_rotvec(
            world_pose[world_valid, :3]
        ).as_matrix()
    return first_frame_aligned_world_joint_error(
        np.concatenate(predicted_joints), np.concatenate(target_joints),
        pred_root.detach().cpu().numpy(), gt_root_rotations, world_valid,
    )


def self_test() -> None:
    identity = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    zero, usable = camera_angular_velocity_with_validity(identity, np.arange(4), np.ones(4, bool))
    assert np.array_equal(zero, np.zeros((3, 6), np.float32)) and usable.all()
    try:
        validate_raw_camera_track(identity, identity.copy(), np.array([0, 1, 2, 3]), np.ones(4, bool))
        mismatch = identity.copy()
        mismatch[1, 0, 3] = 5.0
        validate_raw_camera_track(identity, mismatch, np.array([0, 1, 2, 3]), np.ones(4, bool))
    except ValueError:
        pass
    else:
        raise AssertionError("raw camera mismatch was not caught")
    print("Self-test passed: full-track camera conversion, validity mask, and raw/parsed cross-check", flush=True)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = (
        args.test_parsed, args.wham_checkpoint, args.hmr2s_checkpoint,
        args.yolo26m_weights, args.adapter_checkpoint, args.h36m_joint_regressor,
        args.wham_joint_regressor, args.wham_repo / "lib/models/wham.py",
        args.hmr2s_repo / "4D-Humans/hmr2/models/backbones/vit.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input: " + ", ".join(missing))
    for gender in ("NEUTRAL", "MALE", "FEMALE"):
        if not (args.smpl_model_directory / f"SMPL_{gender}.pkl").is_file():
            raise FileNotFoundError(args.smpl_model_directory / f"SMPL_{gender}.pkl")
    if not torch.cuda.is_available():
        raise RuntimeError("This full evaluation requires a Kaggle CUDA GPU")
    wham_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True).strip()
    hmr_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.hmr2s_repo, text=True).strip()
    if wham_commit != wham_eval.WHAM_COMMIT or hmr_commit != HMR2S_COMMIT:
        raise RuntimeError(f"Unexpected source commits: WHAM={wham_commit}; HMR2-S={hmr_commit}")
    expected_hashes = (
        (args.hmr2s_checkpoint, HMR2S_CHECKPOINT_SHA256),
        (args.yolo26m_weights, args.expected_yolo_sha256),
        (args.adapter_checkpoint, args.expected_adapter_sha256),
    )
    for path, expected in expected_hashes:
        if smoothing.sha256(path) != expected:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {path}")

    device = torch.device("cuda")
    image_root = wham_eval.locate_image_root(args.three_dpw_root)
    sequence_root = locate_sequence_root(args.three_dpw_root)
    labels = joblib.load(args.test_parsed)
    for name in ("vid", "frame_id", "pose", "betas", "gender", "cam_poses"):
        if name not in labels:
            raise KeyError(f"Parsed 3DPW test annotations lack {name}")
    selected = smoothing.test_indices(labels, image_root)
    actual_population = {
        str(labels["vid"][index]): frozen_eval.available_frames(labels, index)
        for index in selected
    }
    if actual_population != EXPECTED_TEST_TRACK_FRAMES:
        raise RuntimeError(
            "The selected one-person test population differs from the prior body-pose evaluation: "
            f"actual={actual_population}; expected={EXPECTED_TEST_TRACK_FRAMES}"
        )
    print(
        f"Locked body-pose test population: {len(selected)} one-person tracks, "
        f"{sum(actual_population.values())} recurrent frames; raw sequence directory: {sequence_root}",
        flush=True,
    )

    network = wham_eval.load_wham_core(args.wham_repo, args.wham_checkpoint, device)
    payload = smoothing.load_payload(args.adapter_checkpoint)
    if not payload.get("uses_learned_adapter"):
        raise RuntimeError("Selected adapter checkpoint is not a learned adapter")
    adapter = smoothing.TokenAdapter().to(device).eval()
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    pose_model = YOLO(str(args.yolo26m_weights))
    initializer = frozen_eval.FrozenSMPLInitializer(
        checkpoint_smpl_buffers(args.hmr2s_checkpoint),
        torch.from_numpy(np.load(args.wham_joint_regressor)).float(),
    ).to(device).eval()
    smpl_models = reference_eval.load_smpl_models(args.smpl_model_directory, device)
    h36m = torch.from_numpy(np.load(args.h36m_joint_regressor)[smoothing.H36M_TO_J14]).float().unsqueeze(0).to(device)

    arrays: dict[str, dict[str, list[np.ndarray]]] = {variant: defaultdict(list) for variant in VARIANTS}
    rows: list[dict[str, Any]] = []
    source_frames = 0
    detections = 0
    camera_motion_magnitudes: list[np.ndarray] = []
    usable_camera_transitions = 0
    valid_world_root_frames = 0
    noncontiguous_transitions = 0
    for ordinal, index in enumerate(selected, 1):
        video_id = str(labels["vid"][index])
        raw, person_index = raw_track(sequence_root, video_id)
        recurrent_frames = frozen_eval.available_frames(labels, index)
        if len(labels["cam_poses"][index]) < recurrent_frames + 1:
            raise ValueError(f"Parsed camera poses shorter than the locked body-pose track {video_id}")
        frame_ids = wham_eval.to_numpy(labels["frame_id"][index])[: recurrent_frames + 1].astype(np.int64)
        parsed_cameras = wham_eval.to_numpy(labels["cam_poses"][index])[: recurrent_frames + 1]
        gap_count = int(np.count_nonzero(np.diff(frame_ids) != 1))
        noncontiguous_transitions += gap_count
        valid_camera = np.asarray(raw["campose_valid"], dtype=bool).reshape(-1)
        if len(valid_camera) != len(raw["cam_poses"]):
            raise ValueError(f"Raw camera-validity count mismatch for {video_id}")
        validate_raw_camera_track(parsed_cameras, raw["cam_poses"], frame_ids, valid_camera)
        person_pose = np.asarray(raw["poses"][person_index], dtype=np.float32)
        person_trans = np.asarray(raw["trans"][person_index], dtype=np.float32)
        if frame_ids.max() >= len(person_pose) or frame_ids.max() >= len(person_trans):
            raise ValueError(f"Raw person trajectory shorter than frame IDs in {video_id}")
        print(
            json.dumps({"stage": "start_track", "track": f"{ordinal}/{len(selected)}", "sequence": video_id, "recurrent_frames": recurrent_frames}),
            flush=True,
        )
        paths = wham_eval.sequence_image_paths(image_root, video_id, frame_ids)
        observations = frozen_eval.frozen_phone_observations(
            pose_model, hmr2s, paths, args.pose_batch_size, args.hmr_batch_size, device
        )
        if observations["valid"][0, 0].item() < 0.5:
            raise RuntimeError(f"Missing initial YOLO detection for {video_id}; refusing to report incomplete full-track coverage")
        source_frames += len(paths)
        detections += int(observations["detections"])
        print(
            json.dumps({"stage": "observations_ready", "track": f"{ordinal}/{len(selected)}", "sequence": video_id, "detections": int(observations["detections"])}),
            flush=True,
        )
        angular, usable = camera_angular_velocity_with_validity(
            parsed_cameras, frame_ids, valid_camera[frame_ids]
        )
        usable_camera_transitions += int(usable.sum())
        camera_motion_magnitudes.append(np.linalg.norm(angular, axis=1))
        oracle_input = torch.from_numpy(angular).unsqueeze(0).to(device)
        zero_input = torch.zeros_like(oracle_input)
        target_pose = torch.from_numpy(wham_eval.to_numpy(labels["pose"][index])[1 : recurrent_frames + 1].astype(np.float32))
        target_betas = torch.from_numpy(wham_eval.to_numpy(labels["betas"][index])[1 : recurrent_frames + 1].astype(np.float32))
        gt_world_pose = person_pose[frame_ids[1:]]
        gt_world_translation = person_trans[frame_ids[1:]]
        world_valid = (
            np.isfinite(gt_world_pose[:, :72]).all(axis=1)
            & np.isfinite(gt_world_translation).all(axis=1)
        )
        valid_world_root_frames += int(world_valid.sum())
        predictions = {
            variant: smoothing.run_phone_variant(
                network, adapter, observations, initializer, device, cam_angvel=camera_input
            )
            for variant, camera_input in zip(VARIANTS, (zero_input, oracle_input))
        }
        for name in ("pose", "shape", "kp3d", "camera", "contact"):
            torch.testing.assert_close(
                predictions[VARIANTS[0]][name], predictions[VARIANTS[1]][name],
                msg=lambda message: f"Camera-motion input changed the {name} body branch: {message}",
            )
        filtered = smoothing.causal_smooth(predictions[VARIANTS[0]], **smoothing.FILTERS["light"])
        pose_metrics, _ = reference_eval.smpl_metrics(
            filtered, target_pose, target_betas, str(labels["gender"][index]).lower(),
            smpl_models, h36m, device, args.smpl_batch_size,
        )
        for name in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm"):
            if len(pose_metrics[name]) != recurrent_frames:
                raise RuntimeError(
                    f"Incomplete {name} coverage on {video_id}: "
                    f"{len(pose_metrics[name])}/{recurrent_frames} frames"
                )
        for variant in VARIANTS:
            prediction = predictions[variant]
            root = root_errors(
                prediction, observations["pose"][0, 0],
                gt_world_pose, gt_world_translation, world_valid,
            )
            world_joints = world_joint_error(
                filtered if variant == VARIANTS[0] else smoothing.causal_smooth(
                    prediction, **smoothing.FILTERS["light"]
                ),
                prediction,
                observations["pose"][0, 0],
                gt_world_pose,
                gt_world_translation,
                target_betas,
                str(labels["gender"][index]).lower(),
                smpl_models,
                h36m,
                world_valid,
                device,
                args.smpl_batch_size,
            )
            row: dict[str, Any] = {
                "variant": variant, "sequence": video_id, "person_index": person_index,
                "first_frame_id": int(frame_ids[0]), "last_frame_id": int(frame_ids[-1]),
                "source_frames": len(paths), "evaluated_frames": len(paths) - 1,
                "noncontiguous_transitions": gap_count,
                "oracle_camera_transitions": int(usable.sum()),
                "world_root_scored_frames": int(world_valid.sum()),
                "detections": int(observations["detections"]),
            }
            for name, values in {
                **root,
                "first_frame_aligned_world_mpjpe_mm": world_joints,
                **pose_metrics,
            }.items():
                if len(values):
                    arrays[variant][name].append(values)
                row[name] = float(np.mean(values)) if len(values) else None
            rows.append(row)
        print(json.dumps({"track": f"{ordinal}/{len(selected)}", "sequence": video_id, "paired_frames": len(paths) - 1, "noncontiguous_transitions": gap_count, "oracle_camera_transitions": int(usable.sum()), "world_root_scored_frames": int(world_valid.sum())}), flush=True)
    expected_frames = sum(EXPECTED_TEST_TRACK_FRAMES.values())
    if len(rows) != 2 * len(selected) or source_frames != expected_frames + len(selected):
        raise RuntimeError("Incomplete coverage of the locked 11-track body-pose population")
    if valid_world_root_frames == 0:
        raise RuntimeError("No valid 3DPW world-root frames were available to score")
    variants = {
        variant: {"metrics": {name: summarize(parts) for name, parts in arrays[variant].items()}}
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        for name in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm"):
            if variants[variant]["metrics"][name]["samples"] != expected_frames:
                raise RuntimeError(f"Incomplete full-11 coverage: {variant} {name}")
        for name in WORLD_METRICS:
            if variants[variant]["metrics"][name]["samples"] != valid_world_root_frames:
                raise RuntimeError(f"Inconsistent world-root coverage: {variant} {name}")
    body_reproduction_delta = {
        name: float(variants[VARIANTS[0]]["metrics"][name]["mean"] - expected)
        for name, expected in EXPECTED_LOCKED_LIGHT_BODY_METRICS.items()
    }
    if any(abs(delta) > BODY_METRIC_REPRODUCTION_TOLERANCE for delta in body_reproduction_delta.values()):
        raise RuntimeError(
            "Body-pose metrics no longer reproduce the locked 11-track report "
            f"within {BODY_METRIC_REPRODUCTION_TOLERANCE} units: {body_reproduction_delta}"
        )
    deltas = {
        name: float(variants[VARIANTS[1]]["metrics"][name]["mean"] - variants[VARIANTS[0]]["metrics"][name]["mean"])
        for name in (*WORLD_METRICS, *POSE_METRICS)
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "locked_iphone_pipeline_3dpw_oracle_camera_motion",
        "protocol": {
            "pipeline": "YOLO26m-pose + released HMR2.0-S + selected token adapter + released WHAM + locked light pose/shape smoothing",
            "comparison": "identical images, detections, HMR2-S tokens, adapter, WHAM weights, and 3DPW test frames; only WHAM cam_angvel changes",
            "zero_variant": "all six camera-angular input values set to zero per recurrent frame",
            "oracle_variant": "3DPW ground-truth camera rotations converted with official WHAM R[t] @ R[t+1].T, first two matrix rows, minus identity 6D, times 30 fps",
            "camera_translation_input": "none; WHAM cam_angvel only accepts the rotational 6D signal",
            "track_selection": "exactly the 11 single-person 3DPW test tracks and all 11,349 recurrent frames used by the locked body-pose evaluation; track IDs and frame counts are asserted",
            "invalid_camera_policy": "run and score the same frames as the locked body-pose evaluation; substitute zero camera input on transitions lacking valid camera poses or adjacent raw frame IDs; camera validity alone does not exclude a world-body label from scoring",
            "pose_smoothing": smoothing.FILTERS["light"],
            "root_diagnostic": "WHAM local velocity integrated with preceding predicted root rotation; predicted root trajectory aligned to raw 3DPW world root orientation and origin at first valid world-root frame, for scoring only",
            "world_joint_diagnostic": "SMPL 14-joint world coordinates from predicted WHAM world root, output-smoothed body pose/shape, and integrated root translation versus raw 3DPW world pose/translation; one first-valid-frame rigid rotation/translation alignment, no scale fit",
            "world_metric_warning": "root and first-frame-aligned world-joint diagnostics are not official WHAM 100-frame W-MPJPE or W-PA-MPJPE and should not be compared numerically to those metrics",
            "camera_relative_metrics": "PA-MPJPE, MPJPE, PVE and acceleration are included as a sanity check; WHAM architecture predicts these on a separate branch and they are expected to be invariant to camera motion",
            "no_training_or_selection": True,
        },
        "population": {
            "requested_one_person_tracks": len(selected),
            "paired_completed_tracks": len(rows) // 2,
            "source_frames": source_frames,
            "paired_evaluated_recurrent_frames": source_frames - len(rows) // 2,
            "detections_on_all_source_frames": detections,
            "oracle_camera_valid_transitions": usable_camera_transitions,
            "oracle_camera_zero_fallback_transitions": expected_frames - usable_camera_transitions,
            "noncontiguous_frame_transitions": noncontiguous_transitions,
            "oracle_camera_transition_coverage": usable_camera_transitions / expected_frames,
            "world_root_scored_frames_per_variant": valid_world_root_frames,
            "world_label_coverage": valid_world_root_frames / expected_frames,
            "nonzero_oracle_camera_frames": int(np.count_nonzero(np.concatenate(camera_motion_magnitudes) > 1e-4)),
            "oracle_6d_camera_input_norm": summarize(camera_motion_magnitudes),
            "skipped_tracks": [],
        },
        "variants": variants,
        "reproduction_of_locked_body_pose_report": {
            "previous_report": "evaluation/results/selected/hmr2s_temporal_smoothing_3dpw.json",
            "expected_light_filter_metric_means": EXPECTED_LOCKED_LIGHT_BODY_METRICS,
            "observed_minus_expected": body_reproduction_delta,
            "absolute_tolerance_per_metric": BODY_METRIC_REPRODUCTION_TOLERANCE,
            "passed": True,
        },
        "oracle_minus_zero_mean": deltas,
        "limitations": [
            "3DPW ground-truth camera rotations are privileged offline labels, unavailable to an iPhone at inference time. This is an oracle-input ablation, not gyro or ARKit accuracy; it is not guaranteed to be an upper bound on pipeline accuracy.",
            "No iPhone sensor noise, timing offset, calibration error, bias drift, or dropout is simulated.",
            "Only camera rotation is injected; no ground-truth camera translation is supplied.",
            "The population intentionally matches the prior 11 single-person body-pose test tracks, not all multi-person 3DPW tracks.",
            "All recurrent frames are processed for body pose. Invalid camera transitions use zero motion in the oracle arm; world-root metrics exclude only non-finite world-body labels, if any, and report their exact coverage.",
            "Parsed 3DPW frame IDs can skip raw video frames. The experiment keeps the prior body-pose frame population and sets oracle camera motion to zero across each gap. World-trajectory drift after a gap may include motion missed by both variants; the gap count is reported.",
            "The first-frame orientation and origin alignment is scoring-only and cannot be treated as an inference calibration step.",
            "This Kaggle GPU run does not measure Core ML or real-device latency and cannot establish iPhone performance.",
        ],
        "provenance": {
            "wham_commit": wham_commit,
            "hmr2s_commit": hmr_commit,
            "wham_checkpoint_sha256": smoothing.sha256(args.wham_checkpoint),
            "hmr2s_checkpoint_sha256": smoothing.sha256(args.hmr2s_checkpoint),
            "yolo26m_checkpoint_sha256": smoothing.sha256(args.yolo26m_weights),
            "adapter_checkpoint_sha256": smoothing.sha256(args.adapter_checkpoint),
            "parsed_3dpw_test_sha256": smoothing.sha256(args.test_parsed),
        },
    }
    return report, rows


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report, rows = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"population": report["population"], "oracle_minus_zero_mean": report["oracle_minus_zero_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
