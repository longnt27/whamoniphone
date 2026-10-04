#!/usr/bin/env python3
"""Paired full-track 3DPW accuracy for released WHAM and the iPhone pipeline.

Both arms receive the same 3DPW-oracle camera rotation and are scored on the
same 11 person tracks. The phone arm executes the same causal SMPL/contact
world step exported to Core ML; the original arm uses that same scorer and
world step with the released WHAM predictions. No ground-truth body pose or
root orientation is used as an inference initializer.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_frozen_hmr2s_wham as frozen_eval
import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_hmr2s_camera_oracle as oracle
import evaluate_hmr2s_temporal_smoothing as smoothing
import evaluate_wham_feature_substitution as wham_eval
import joblib
import numpy as np
import torch
from camera_motion_oracle import (
    camera_angular_velocity_with_validity,
    first_frame_aligned_world_joint_error,
    masked_first_valid_root_errors,
)
from export_wham_world_step import WHAMWorldStep
from hmr2s_frozen import HMR2S_CHECKPOINT_SHA256, HMR2S_COMMIT, FrozenHMR2S, checkpoint_smpl_buffers
from paired_world_contract import verify_paired_population
from scipy.spatial.transform import Rotation
from ultralytics import YOLO


ARMS = ("released_wham", "iphone_pipeline")
WORLD_METRICS = (
    "root_orientation_error_deg",
    "root_displacement_error_mm",
    "first_frame_aligned_world_mpjpe_mm",
)
EXPECTED_ORIGINAL_BODY = {
    "pa_mpjpe_mm": 32.66937125786904,
    "mpjpe_mm": 55.575809034736935,
    "pve_mm": 65.63646479835279,
    "accel_official_30fps": 6.226210768185375,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in (
        "test-parsed", "three-dpw-root", "wham-repo", "wham-checkpoint",
        "hmr2s-repo", "hmr2s-checkpoint", "yolo26m-weights", "adapter-checkpoint",
        "smpl-model-directory", "h36m-joint-regressor", "wham-joint-regressor",
        "feet-regressor", "output-json", "output-csv",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-yolo-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--pose-batch-size", type=int, default=24)
    parser.add_argument("--hmr-batch-size", type=int, default=24)
    parser.add_argument("--smpl-batch-size", type=int, default=192)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    expected = {"a": 3, "b": 2}
    assert verify_paired_population(expected, expected.copy(), expected.copy()) == 5
    try:
        verify_paired_population(expected, expected.copy(), {"a": 3, "b": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("A shortened phone track was accepted")
    oracle.self_test()
    print("Self-test passed: paired coverage, camera input, and raw 3DPW checks", flush=True)


def original_prediction(
    network: torch.nn.Module,
    labels: dict[str, Any],
    index: int,
    frames: int,
    angular: torch.Tensor,
    device: torch.device,
    wham_repo: Path,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Official cached ViTPose/HMR2 inputs and flip averaging, without GT root."""

    normal = reference_eval.official_inputs(
        labels, index, frames, "", device, use_target_first_root=False
    )
    flipped = reference_eval.official_inputs(
        labels, index, frames, "flipped_", device, use_target_first_root=False
    )
    for inputs in (normal, flipped):
        # The scorer sees labels, but the world-space network initializer does not.
        inputs["cam_angvel"] = angular
    flipped_output = reference_eval.run_core(network, flipped)
    normal_output = reference_eval.run_core(network, normal)
    averaged = reference_eval.average_flipped_prediction(wham_repo, normal_output, flipped_output)
    combined = dict(normal_output)
    combined["pose"] = averaged["pose"]
    combined["shape"] = averaged["shape"]
    # The prior released-WHAM body benchmark flip-averaged only pose/shape.
    # Keep the normal branch's contact/trajectory/context together; averaging
    # flipped contact here would invent a new world variant.
    combined["contact"] = normal_output["contact"]
    return combined, normal["init_root"]


def make_world_step(
    network: torch.nn.Module,
    hmr2s_checkpoint: Path,
    wham_regressor: torch.Tensor,
    feet_regressor: torch.Tensor,
    device: torch.device,
) -> WHAMWorldStep:
    if tuple(feet_regressor.shape) != (4, 6890):
        raise ValueError("The official WHAM foot regressor must be 4x6890")
    step = WHAMWorldStep(
        network.trajectory_refiner,
        checkpoint_smpl_buffers(hmr2s_checkpoint),
        wham_regressor,
        feet_regressor,
    ).to(device).eval()
    return step


@torch.inference_mode()
def run_world_step(
    step: WHAMWorldStep,
    prediction: dict[str, torch.Tensor],
    initial_root: torch.Tensor,
    h36m_regressor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the exported causal contact/refiner step for every frame."""

    frames = int(prediction["pose"].shape[1])
    if any(int(prediction[name].shape[1]) != frames for name in
           ("shape", "contact", "root", "velocity", "kp3d", "motion_context")):
        raise ValueError("World-step inputs have unequal frame counts")
    device = prediction["pose"].device
    root = initial_root.reshape(1, 1, 6).float()
    zero_translation = torch.zeros(1, 1, 3, device=device)
    zero_feet = torch.zeros(1, 1, 4, 3, device=device)
    h_refiner = torch.zeros(2, 1, 512, device=device)
    c_refiner = torch.zeros_like(h_refiner)
    prev_root, prev_refined_root = root, root
    prev_translation, prev_refined_translation = zero_translation, zero_translation
    prev_body_feet, prev_world_feet = zero_feet, zero_feet
    predicted_joints: list[np.ndarray] = []
    predicted_roots: list[np.ndarray] = []
    predicted_translations: list[np.ndarray] = []
    for frame in range(frames):
        h_enc = torch.zeros(3, 1, 512, device=device)
        h_enc[-1, 0] = prediction["motion_context"][0, frame, :512]
        values = step(
            prediction["pose"][:, frame:frame + 1].reshape(1, 1, 144),
            prediction["shape"][:, frame:frame + 1],
            prediction["contact"][:, frame:frame + 1],
            prediction["root"][:, frame:frame + 1],
            prediction["velocity"][:, frame:frame + 1],
            prediction["kp3d"][:, frame:frame + 1].reshape(1, 1, 51),
            h_enc,
            prev_root, prev_translation, prev_body_feet, prev_world_feet,
            prev_refined_root, prev_refined_translation,
            h_refiner, c_refiner,
            torch.tensor(float(frame > 0), device=device),
        )
        vertices = values[0].reshape(1, 6890, 3)
        joints = torch.matmul(h36m_regressor, vertices)
        predicted_joints.append(joints[0].detach().cpu().numpy())
        predicted_roots.append(
            wham_eval.rotation_6d_to_matrix(values[2].reshape(1, 6))[0].detach().cpu().numpy()
        )
        predicted_translations.append(values[4].reshape(3).detach().cpu().numpy())
        prev_refined_root, prev_refined_translation = values[2], values[4]
        prev_root, prev_translation = values[5], values[6]
        prev_body_feet, prev_world_feet = values[7], values[8]
        h_refiner, c_refiner = values[9], values[10]
    return (
        np.stack(predicted_joints),
        np.stack(predicted_roots),
        np.stack(predicted_translations),
    )


@torch.inference_mode()
def target_world_joints(
    world_pose: np.ndarray,
    world_translation: np.ndarray,
    target_betas: torch.Tensor,
    gender: str,
    models: dict[str, torch.nn.Module],
    wham_regressor: torch.Tensor,
    h36m_regressor: torch.Tensor,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    safe_pose = np.asarray(world_pose[:, :72], dtype=np.float32).copy()
    safe_translation = np.asarray(world_translation, dtype=np.float32).copy()
    safe_pose[~valid] = 0
    safe_translation[~valid] = 0
    matrices = wham_eval.axis_angle_to_matrix(
        torch.from_numpy(safe_pose).reshape(-1, 24, 3).to(device)
    )
    joints: list[np.ndarray] = []
    for start in range(0, len(matrices), batch_size):
        stop = min(start + batch_size, len(matrices))
        output = models[gender](
            body_pose=matrices[start:stop, 1:],
            global_orient=matrices[start:stop, :1],
            betas=target_betas[start:stop].to(device),
            pose2rot=False,
        )
        verts = output.vertices
        wham_joints = torch.matmul(wham_regressor, verts)
        pelvis = wham_joints[:, [11, 12]].mean(dim=1, keepdim=True)
        centered = verts - pelvis + torch.from_numpy(safe_translation[start:stop]).to(device)[:, None]
        joints.append(torch.matmul(h36m_regressor, centered).detach().cpu().numpy())
    roots = np.repeat(np.eye(3, dtype=np.float32)[None], len(matrices), axis=0)
    if valid.any():
        roots[valid] = Rotation.from_rotvec(world_pose[valid, :3]).as_matrix()
    return np.concatenate(joints), roots


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = (
        args.test_parsed, args.wham_checkpoint, args.hmr2s_checkpoint,
        args.yolo26m_weights, args.adapter_checkpoint, args.h36m_joint_regressor,
        args.wham_joint_regressor, args.feet_regressor,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input: " + ", ".join(missing))
    for gender in ("NEUTRAL", "MALE", "FEMALE"):
        if not (args.smpl_model_directory / f"SMPL_{gender}.pkl").is_file():
            raise FileNotFoundError(f"Missing licensed SMPL_{gender}.pkl")
    if not torch.cuda.is_available():
        raise RuntimeError("This complete comparison requires a Kaggle CUDA GPU")
    for repo, commit in ((args.wham_repo, wham_eval.WHAM_COMMIT), (args.hmr2s_repo, HMR2S_COMMIT)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Unexpected source commit {repo}: {actual}")
    for path, expected in (
        (args.hmr2s_checkpoint, HMR2S_CHECKPOINT_SHA256),
        (args.yolo26m_weights, args.expected_yolo_sha256),
        (args.adapter_checkpoint, args.expected_adapter_sha256),
    ):
        if smoothing.sha256(path) != expected:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")

    device = torch.device("cuda")
    image_root = wham_eval.locate_image_root(args.three_dpw_root)
    sequence_root = oracle.locate_sequence_root(args.three_dpw_root)
    labels = joblib.load(args.test_parsed)
    for name in (
        "vid", "frame_id", "pose", "betas", "gender", "cam_poses",
        "kp2d", "bbox", "features", "init_kp3d", "init_pose",
        "flipped_kp2d", "flipped_bbox", "flipped_features",
        "flipped_init_kp3d", "flipped_init_pose",
    ):
        if name not in labels:
            raise KeyError(f"Parsed 3DPW test lacks {name}")
    selected = smoothing.test_indices(labels, image_root)
    population = {
        str(labels["vid"][index]): frozen_eval.available_frames(labels, index)
        for index in selected
    }
    expected_frames = verify_paired_population(
        oracle.EXPECTED_TEST_TRACK_FRAMES, population, population
    )
    print(f"Locked paired test: {len(selected)} tracks, {expected_frames} recurrent frames", flush=True)

    network = wham_eval.load_wham_core(args.wham_repo, args.wham_checkpoint, device)
    adapter_payload = smoothing.load_payload(args.adapter_checkpoint)
    if not adapter_payload.get("uses_learned_adapter"):
        raise RuntimeError("The selected adapter is not a learned adapter")
    adapter = smoothing.TokenAdapter().to(device).eval()
    adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    pose_model = YOLO(str(args.yolo26m_weights))
    initializer = frozen_eval.FrozenSMPLInitializer(
        checkpoint_smpl_buffers(args.hmr2s_checkpoint),
        torch.from_numpy(np.load(args.wham_joint_regressor)).float(),
    ).to(device).eval()
    models = reference_eval.load_smpl_models(args.smpl_model_directory, device)
    wham_regressor = torch.from_numpy(np.load(args.wham_joint_regressor)).float().to(device)
    feet_regressor = torch.from_numpy(np.load(args.feet_regressor)).float().to(device)
    h36m_regressor = torch.from_numpy(
        np.load(args.h36m_joint_regressor)[smoothing.H36M_TO_J14]
    ).float().to(device)
    world_step = make_world_step(
        network, args.hmr2s_checkpoint, wham_regressor, feet_regressor, device
    )
    arrays: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    rows: list[dict[str, Any]] = []
    completed = {arm: {} for arm in ARMS}
    source_frames = detections = camera_transitions = gap_transitions = world_frames = 0
    for ordinal, index in enumerate(selected, 1):
        video_id = str(labels["vid"][index])
        raw, person_index = oracle.raw_track(sequence_root, video_id)
        count = population[video_id]
        ids = wham_eval.to_numpy(labels["frame_id"][index])[: count + 1].astype(np.int64)
        parsed_camera = wham_eval.to_numpy(labels["cam_poses"][index])[: count + 1]
        valid_camera = np.asarray(raw["campose_valid"], dtype=bool).reshape(-1)
        oracle.validate_raw_camera_track(parsed_camera, raw["cam_poses"], ids, valid_camera)
        paths = wham_eval.sequence_image_paths(image_root, video_id, ids)
        observations = frozen_eval.frozen_phone_observations(
            pose_model, hmr2s, paths, args.pose_batch_size, args.hmr_batch_size, device
        )
        if observations["valid"][0, 0].item() < 0.5:
            raise RuntimeError(f"Missing initializer detection in {video_id}")
        source_frames += len(paths)
        detections += int(observations["detections"])
        angular, usable = camera_angular_velocity_with_validity(
            parsed_camera, ids, valid_camera[ids]
        )
        camera_transitions += int(usable.sum())
        gaps = int(np.count_nonzero(np.diff(ids) != 1))
        gap_transitions += gaps
        angular_tensor = torch.from_numpy(angular).unsqueeze(0).to(device)
        original, original_root = original_prediction(
            network, labels, index, count, angular_tensor, device, args.wham_repo
        )
        phone = smoothing.run_phone_variant(
            network, adapter, observations, initializer, device, cam_angvel=angular_tensor
        )
        phone_filtered = smoothing.causal_smooth(phone, **smoothing.FILTERS["light"])
        phone_root = observations["pose"][:1, 0].reshape(1, 1, 6).to(device)
        predictions = {
            "released_wham": (original, original_root),
            "iphone_pipeline": (phone_filtered, phone_root),
        }
        target_pose = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1:count + 1].astype(np.float32)
        )
        target_betas = torch.from_numpy(
            wham_eval.to_numpy(labels["betas"][index])[1:count + 1].astype(np.float32)
        )
        world_pose = np.asarray(raw["poses"][person_index], dtype=np.float32)[ids[1:]]
        world_translation = np.asarray(raw["trans"][person_index], dtype=np.float32)[ids[1:]]
        valid_world = np.isfinite(world_pose[:, :72]).all(axis=1) & np.isfinite(world_translation).all(axis=1)
        world_frames += int(valid_world.sum())
        target_joints, target_roots = target_world_joints(
            world_pose, world_translation, target_betas,
            str(labels["gender"][index]).lower(), models,
            wham_regressor, h36m_regressor, valid_world, device, args.smpl_batch_size,
        )
        for arm, (prediction, initial_root) in predictions.items():
            body_metrics, _ = reference_eval.smpl_metrics(
                prediction, target_pose, target_betas,
                str(labels["gender"][index]).lower(), models,
                h36m_regressor.unsqueeze(0), device, args.smpl_batch_size,
            )
            pred_joints, pred_roots, pred_translation = run_world_step(
                world_step, prediction, initial_root, h36m_regressor
            )
            root_orientation, root_displacement = masked_first_valid_root_errors(
                pred_roots, pred_translation, target_roots, world_translation, valid_world
            )
            world_error = first_frame_aligned_world_joint_error(
                pred_joints, target_joints, pred_roots, target_roots, valid_world
            )
            measurements = {
                **body_metrics,
                "root_orientation_error_deg": root_orientation,
                "root_displacement_error_mm": root_displacement,
                "first_frame_aligned_world_mpjpe_mm": world_error,
            }
            row: dict[str, Any] = {
                "arm": arm, "sequence": video_id, "person_index": person_index,
                "source_frames": len(paths), "evaluated_frames": count,
                "world_scored_frames": int(valid_world.sum()),
                "oracle_camera_transitions": int(usable.sum()),
                "noncontiguous_transitions": gaps,
                "detections": int(observations["detections"]),
            }
            for metric, values in measurements.items():
                if len(values):
                    arrays[arm][metric].append(np.asarray(values))
                row[metric] = float(np.mean(values)) if len(values) else None
            rows.append(row)
            completed[arm][video_id] = count
        print(json.dumps({"track": f"{ordinal}/{len(selected)}", "sequence": video_id, "paired_frames": count}), flush=True)
    verify_paired_population(oracle.EXPECTED_TEST_TRACK_FRAMES, *[completed[arm] for arm in ARMS])
    if source_frames != expected_frames + len(selected):
        raise RuntimeError("Source frame coverage differs from locked body-pose population")
    variants = {
        arm: {"metrics": {name: oracle.summarize(parts) for name, parts in arrays[arm].items()}}
        for arm in ARMS
    }
    for arm in ARMS:
        for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm"):
            if variants[arm]["metrics"][metric]["samples"] != expected_frames:
                raise RuntimeError(f"Incomplete paired body metric: {arm} {metric}")
        for metric in WORLD_METRICS:
            if variants[arm]["metrics"][metric]["samples"] != world_frames:
                raise RuntimeError(f"Incomplete paired world metric: {arm} {metric}")
    expected = {
        "released_wham": EXPECTED_ORIGINAL_BODY,
        "iphone_pipeline": oracle.EXPECTED_LOCKED_LIGHT_BODY_METRICS,
    }
    reproduction = {}
    for arm in ARMS:
        drift = {
            metric: variants[arm]["metrics"][metric]["mean"] - mean
            for metric, mean in expected[arm].items()
        }
        reproduction[arm] = {"observed_minus_locked": drift, "passed": all(abs(value) <= 2.0 for value in drift.values())}
    if not all(item["passed"] for item in reproduction.values()):
        raise RuntimeError(f"Body-pose reproduction failed; refusing world comparison: {reproduction}")
    deltas = {
        metric: variants["iphone_pipeline"]["metrics"][metric]["mean"]
        - variants["released_wham"]["metrics"][metric]["mean"]
        for metric in variants["released_wham"]["metrics"]
    }
    report = {
        "schema_version": 1,
        "experiment": "released_wham_vs_iphone_pipeline_paired_full_3dpw",
        "model_provenance": {
            "wham_commit": wham_eval.WHAM_COMMIT,
            "hmr2s_commit": HMR2S_COMMIT,
            "hmr2s_checkpoint_sha256": HMR2S_CHECKPOINT_SHA256,
            "yolo26m_checkpoint_sha256": args.expected_yolo_sha256,
            "selected_adapter_sha256": args.expected_adapter_sha256,
            "world_step_source": "export_wham_world_step.WHAMWorldStep",
        },
        "protocol": {
            "dataset": "same locked 11 single-person 3DPW test tracks and all recurrent frames",
            "camera_input": "same 3DPW ground-truth camera rotation converted to WHAM 6D angular input for both arms; zero only at invalid/gapped transitions",
            "original": "official stored ViTPose/HMR2 inputs, flip-averaged pose/shape, normal-branch contact/trajectory/context, released WHAM; first root from HMR2 initializer, not test label",
            "phone": "YOLO26m-pose, HMR2.0-S, selected token adapter, released WHAM, light output smoothing",
            "world_stage": "identical exported iPhone causal SMPL/contact-aware WHAMWorldStep weights and recurrence for both arms",
            "world_score": "same 14 SMPL-regressed joints, one rigid first-frame rotation/translation alignment per full track, no scale fit or 100-frame reset",
            "world_score_warning": "custom paired full-track diagnostic, not the WHAM paper's EMDB-2 W-MPJPE100",
            "no_training_or_model_selection": True,
        },
        "population": {
            "tracks_per_arm": len(selected),
            "recurrent_frames_per_arm": expected_frames,
            "source_frames": source_frames,
            "phone_detections_on_source_frames": detections,
            "world_scored_frames_per_arm": world_frames,
            "world_label_coverage": world_frames / expected_frames,
            "valid_oracle_camera_transitions": camera_transitions,
            "zero_camera_fallback_transitions": expected_frames - camera_transitions,
            "noncontiguous_frame_transitions": gap_transitions,
        },
        "reproduction_of_locked_body_pose": reproduction,
        "variants": variants,
        "iphone_minus_original_mean": deltas,
        "limitations": [
            "3DPW ground-truth camera motion is an offline oracle supplied to both arms; this does not measure iPhone gyro/ARKit error.",
            "The original frontend uses official cached test inputs, not a timed fresh HMR2/ViTPose execution.",
            "The original network's world predictions are passed through the same PyTorch implementation of the iPhone-exported world step, rather than a separate native WHAM application runner.",
            "Nonadjacent frames are not imputed; both arms receive a zero camera-motion transition at each gap and continue recurrent state across it.",
            "The iPhone branch runs its equivalent PyTorch stages on Kaggle, not compiled Core ML packages; conversion numerical differences are not measured.",
            "This Kaggle run measures accuracy, not Core ML or device latency.",
            "The subset is exactly the same 11 single-person tracks as prior body-pose tests, not every person in 3DPW test.",
        ],
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
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"population": report["population"], "iphone_minus_original_mean": report["iphone_minus_original_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
