"""Select and test causal output smoothing for the locked mobile WHAM pipeline."""

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

import evaluate_frozen_hmr2s_wham as frozen_eval
import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_wham_feature_substitution as wham_eval
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from hmr2s_frozen import (
    HMR2S_CHECKPOINT_SHA256,
    HMR2S_COMMIT,
    FrozenHMR2S,
    checkpoint_smpl_buffers,
)
from scipy.spatial.transform import Rotation
from torch import nn
from ultralytics import YOLO

SCHEMA_VERSION = 1
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
FILTERS: dict[str, dict[str, float]] = {
    "raw": {"pose_alpha": 1.0, "shape_alpha": 1.0},
    "light": {"pose_alpha": 0.75, "shape_alpha": 0.35},
    "medium": {"pose_alpha": 0.55, "shape_alpha": 0.20},
    "strong": {"pose_alpha": 0.35, "shape_alpha": 0.10},
}
METRICS = (
    "pa_mpjpe_mm",
    "mpjpe_mm",
    "pve_mm",
    "accel_official_30fps",
)


class TokenAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.in_projection = nn.Linear(1024, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, 1024)
        self.register_buffer("target_mean", torch.zeros(1024))
        self.register_buffer("target_std", torch.ones(1024))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        correction = self.out_projection(F.gelu(self.in_projection(self.norm(token))))
        return token + correction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--val-parsed", type=Path, required=True)
    parser.add_argument("--test-parsed", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr2s-repo", type=Path, required=True)
    parser.add_argument("--hmr2s-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo26m-weights", type=Path, required=True)
    parser.add_argument("--expected-yolo-sha256", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--smpl-model-directory", type=Path, required=True)
    parser.add_argument("--h36m-joint-regressor", type=Path, required=True)
    parser.add_argument("--wham-joint-regressor", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--validation-tracks", type=int, default=12)
    parser.add_argument("--validation-frames", type=int, default=300)
    parser.add_argument("--pose-batch-size", type=int, default=24)
    parser.add_argument("--hmr-batch-size", type=int, default=24)
    parser.add_argument("--smpl-batch-size", type=int, default=192)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def require_files(args: argparse.Namespace) -> None:
    paths = (
        args.val_parsed,
        args.test_parsed,
        args.wham_checkpoint,
        args.hmr2s_checkpoint,
        args.yolo26m_weights,
        args.adapter_checkpoint,
        args.h36m_joint_regressor,
        args.wham_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
        args.hmr2s_repo / "4D-Humans/hmr2/models/backbones/vit.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing smoothing-evaluation inputs: " + ", ".join(missing)
        )
    for gender in ("NEUTRAL", "MALE", "FEMALE"):
        path = args.smpl_model_directory / f"SMPL_{gender}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)


def validation_indices(
    labels: dict[str, Any], image_root: Path, maximum: int
) -> list[int]:
    bases = [str(value).rsplit("_", 1)[0] for value in labels["vid"]]
    counts = Counter(bases)
    candidates = [
        index
        for index, base in enumerate(bases)
        if counts[base] == 1 and (image_root / base).is_dir()
    ]
    if not candidates:
        candidates = [
            index for index, base in enumerate(bases) if (image_root / base).is_dir()
        ]
    if maximum > 0 and len(candidates) > maximum:
        positions = np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)
        candidates = [candidates[int(position)] for position in positions]
    if not candidates:
        raise RuntimeError("No 3DPW validation track matched raw images")
    return candidates


def test_indices(labels: dict[str, Any], image_root: Path) -> list[int]:
    matching = [
        index
        for index, raw_video in enumerate(labels["vid"])
        if (image_root / str(raw_video).rsplit("_", 1)[0]).is_dir()
    ]
    counts = Counter(str(labels["vid"][index]).rsplit("_", 1)[0] for index in matching)
    result = [
        index
        for index in matching
        if counts[str(labels["vid"][index]).rsplit("_", 1)[0]] == 1
    ]
    if not result:
        raise RuntimeError("No single-person 3DPW test track matched raw images")
    return result


@torch.inference_mode()
def run_phone_variant(
    network: nn.Module,
    adapter: TokenAdapter,
    observations: dict[str, Any],
    initializer: frozen_eval.FrozenSMPLInitializer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    frames = int(observations["frames"]) - 1
    first_pose = observations["pose"][:1].to(device)
    first_betas = observations["betas"][:1].to(device)
    init_joints = initializer(first_pose, first_betas).reshape(1, 1, 51)
    token = adapter(observations["token"][1:].to(device))
    return frozen_eval.run_phone_core(
        network=network,
        x=observations["x"][1:].unsqueeze(0).to(device),
        mask=observations["mask"][1:].unsqueeze(0).to(device),
        features=token.unsqueeze(0),
        feature_valid=observations["valid"][1:].unsqueeze(0).to(device),
        init_kp=torch.cat(
            (init_joints, observations["x"][:1].reshape(1, 1, 37).to(device)), dim=-1
        ),
        init_pose=first_pose.reshape(1, 1, 24, 6),
        init_root=first_pose[:, 0].reshape(1, 1, 6),
        cam_angvel=torch.zeros(1, frames, 6, device=device),
    )


def causal_smooth(
    prediction: dict[str, torch.Tensor], pose_alpha: float, shape_alpha: float
) -> dict[str, torch.Tensor]:
    if pose_alpha >= 1.0 and shape_alpha >= 1.0:
        return prediction
    device = prediction["pose"].device
    dtype = prediction["pose"].dtype
    pose = prediction["pose"].detach().cpu().reshape(-1, 24, 6)
    matrices = wham_eval.rotation_6d_to_matrix(pose).numpy()
    filtered = np.empty_like(matrices)
    filtered[0] = matrices[0]
    previous = Rotation.from_matrix(matrices[0])
    for frame in range(1, len(matrices)):
        current = Rotation.from_matrix(matrices[frame])
        relative = previous.inv() * current
        previous = previous * Rotation.from_rotvec(relative.as_rotvec() * pose_alpha)
        filtered[frame] = previous.as_matrix()
    filtered_pose = wham_eval.matrix_to_rotation_6d(
        torch.from_numpy(filtered)
    ).reshape_as(prediction["pose"])

    shape = prediction["shape"].detach().cpu().clone().reshape(-1, 10)
    for frame in range(1, len(shape)):
        shape[frame] = (
            shape_alpha * shape[frame] + (1.0 - shape_alpha) * shape[frame - 1]
        )
    result = dict(prediction)
    result["pose"] = filtered_pose.to(device=device, dtype=dtype)
    result["shape"] = shape.reshape_as(prediction["shape"]).to(
        device=device, dtype=prediction["shape"].dtype
    )
    return result


def summarize(parts: list[np.ndarray]) -> dict[str, float | int]:
    return frozen_eval.concatenate_summary(parts)


def mean_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {name: float(result["metrics"][name]["mean"]) for name in METRICS}


def score_against_raw(candidate: dict[str, Any], raw: dict[str, Any]) -> float:
    weights = {
        "pa_mpjpe_mm": 0.40,
        "mpjpe_mm": 0.25,
        "pve_mm": 0.25,
        "accel_official_30fps": 0.10,
    }
    return sum(
        weight
        * float(candidate["metrics"][name]["mean"])
        / max(float(raw["metrics"][name]["mean"]), 1e-8)
        for name, weight in weights.items()
    )


@torch.inference_mode()
def evaluate_split(
    split: str,
    labels: dict[str, Any],
    indices: list[int],
    maximum_frames: int,
    filter_names: tuple[str, ...],
    image_root: Path,
    network: nn.Module,
    adapter: TokenAdapter,
    pose_model: YOLO,
    hmr2s: FrozenHMR2S,
    initializer: frozen_eval.FrozenSMPLInitializer,
    smpl_models: dict[str, nn.Module],
    h36m_regressor: torch.Tensor,
    device: torch.device,
    pose_batch_size: int,
    hmr_batch_size: int,
    smpl_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in filter_names
    }
    filter_seconds = defaultdict(float)
    rows: list[dict[str, Any]] = []
    source_frames = 0
    detections = 0
    skipped: list[dict[str, str]] = []
    for ordinal, index in enumerate(indices, start=1):
        video_id = str(labels["vid"][index])
        frames = frozen_eval.available_frames(labels, index)
        if maximum_frames > 0:
            frames = min(frames, maximum_frames)
        if frames <= 0:
            skipped.append({"sequence": video_id, "reason": "no recurrent frames"})
            continue
        frame_ids = wham_eval.to_numpy(labels["frame_id"][index])
        paths = wham_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: frames + 1]
        )
        observations = frozen_eval.frozen_phone_observations(
            pose_model,
            hmr2s,
            paths,
            pose_batch_size,
            hmr_batch_size,
            device,
        )
        source_frames += int(observations["frames"])
        detections += int(observations["detections"])
        if observations["valid"][0, 0].item() < 0.5:
            skipped.append({"sequence": video_id, "reason": "no initializer detection"})
            continue
        raw_prediction = run_phone_variant(
            network, adapter, observations, initializer, device
        )
        target_pose = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1 : frames + 1].astype(np.float32)
        )
        target_betas = torch.from_numpy(
            wham_eval.to_numpy(labels["betas"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        for filter_name in filter_names:
            parameters = FILTERS[filter_name]
            started = time.perf_counter()
            prediction = causal_smooth(raw_prediction, **parameters)
            filter_seconds[filter_name] += time.perf_counter() - started
            metrics, _ = reference_eval.smpl_metrics(
                prediction,
                target_pose,
                target_betas,
                str(labels["gender"][index]).lower(),
                smpl_models,
                h36m_regressor,
                device,
                smpl_batch_size,
            )
            row: dict[str, Any] = {
                "split": split,
                "filter": filter_name,
                "sequence": video_id,
                "frames": frames,
                "detections": int(observations["detections"]),
                "detection_rate": int(observations["detections"]) / len(paths),
            }
            for metric, values in metrics.items():
                accumulated[filter_name][metric].append(values)
                row[metric] = float(values.mean()) if len(values) else None
            rows.append(row)
        print(
            json.dumps(
                {
                    "stage": f"{split}_smoothing",
                    "track": f"{ordinal}/{len(indices)}",
                    "sequence": video_id,
                    "frames": frames,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not rows:
        raise RuntimeError(f"No {split} track completed smoothing evaluation")
    variants = {
        name: {
            "parameters": FILTERS[name],
            "metrics": {
                metric: summarize(parts) for metric, parts in accumulated[name].items()
            },
            "filter_seconds_python_scipy": filter_seconds[name],
        }
        for name in filter_names
    }
    return (
        {
            "variants": variants,
            "population": {
                "requested_tracks": len(indices),
                "completed_tracks": len({row["sequence"] for row in rows}),
                "source_frames": source_frames,
                "detections": detections,
                "detection_rate": detections / max(source_frames, 1),
                "skipped": skipped,
            },
        },
        rows,
    )


def self_test() -> None:
    identity = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 5, 24, 1, 1)
    pose = wham_eval.matrix_to_rotation_6d(identity)
    prediction = {"pose": pose, "shape": torch.arange(50).reshape(1, 5, 10).float()}
    raw = causal_smooth(prediction, 1.0, 1.0)
    assert raw is prediction
    filtered = causal_smooth(prediction, 0.5, 0.5)
    recovered = wham_eval.rotation_6d_to_matrix(filtered["pose"])
    assert torch.allclose(recovered, identity)
    assert torch.allclose(
        filtered["shape"][0, 1],
        torch.arange(10, 20).float() * 0.5 + torch.arange(10).float() * 0.5,
    )
    assert (
        abs(
            score_against_raw(
                {"metrics": {name: {"mean": 5.0} for name in METRICS}},
                {"metrics": {name: {"mean": 10.0} for name in METRICS}},
            )
            - 0.5
        )
        < 1e-8
    )
    print("Self-test passed: rotation smoothing, shape smoothing, and selection score")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    args = parse_args()
    if args.self_test:
        self_test()
        return
    require_files(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this evaluation")
    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    hmr2s_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.hmr2s_repo, text=True
    ).strip()
    if wham_commit != wham_eval.WHAM_COMMIT:
        raise RuntimeError(f"Wrong WHAM commit: {wham_commit}")
    if hmr2s_commit != HMR2S_COMMIT:
        raise RuntimeError(f"Wrong HMR2-S commit: {hmr2s_commit}")
    if sha256(args.hmr2s_checkpoint) != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError("Wrong HMR2-S checkpoint")
    if sha256(args.yolo26m_weights) != args.expected_yolo_sha256:
        raise RuntimeError("Wrong YOLO26m-pose checkpoint")
    if sha256(args.adapter_checkpoint) != args.expected_adapter_sha256:
        raise RuntimeError("Wrong validation-selected adapter checkpoint")

    device = torch.device("cuda")
    network = wham_eval.load_wham_core(args.wham_repo, args.wham_checkpoint, device)
    adapter_payload = load_payload(args.adapter_checkpoint)
    if not adapter_payload.get("uses_learned_adapter"):
        raise RuntimeError(
            "The selected checkpoint does not enable its learned adapter"
        )
    adapter = TokenAdapter().to(device)
    adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    adapter.eval()
    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    pose_model = YOLO(str(args.yolo26m_weights))
    initializer = (
        frozen_eval.FrozenSMPLInitializer(
            checkpoint_smpl_buffers(args.hmr2s_checkpoint),
            torch.from_numpy(np.load(args.wham_joint_regressor)).float(),
        )
        .to(device)
        .eval()
    )
    smpl_models = reference_eval.load_smpl_models(args.smpl_model_directory, device)
    h36m_regressor = (
        torch.from_numpy(np.load(args.h36m_joint_regressor)[H36M_TO_J14])
        .float()
        .unsqueeze(0)
        .to(device)
    )
    image_root = wham_eval.locate_image_root(args.three_dpw_root)

    validation_labels = joblib.load(args.val_parsed)
    validation, rows = evaluate_split(
        "validation",
        validation_labels,
        validation_indices(validation_labels, image_root, args.validation_tracks),
        args.validation_frames,
        tuple(FILTERS),
        image_root,
        network,
        adapter,
        pose_model,
        hmr2s,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        args.pose_batch_size,
        args.hmr_batch_size,
        args.smpl_batch_size,
    )
    raw_validation = validation["variants"]["raw"]
    for result in validation["variants"].values():
        result["score_vs_raw"] = score_against_raw(result, raw_validation)
        result["spatial_gate_passed"] = all(
            float(result["metrics"][metric]["mean"])
            <= 1.05 * float(raw_validation["metrics"][metric]["mean"])
            for metric in METRICS[:3]
        )
    eligible = [
        (name, result)
        for name, result in validation["variants"].items()
        if result["spatial_gate_passed"]
    ]
    selected_name, selected_validation = min(
        eligible,
        key=lambda item: (
            item[1]["score_vs_raw"],
            list(FILTERS).index(item[0]),
        ),
    )
    if selected_validation["score_vs_raw"] >= 1.0:
        selected_name = "raw"
        selected_validation = raw_validation
    print(
        json.dumps(
            {
                "validation_locked_filter": selected_name,
                "validation_score_vs_raw": selected_validation["score_vs_raw"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Protocol boundary: test labels are first opened after the filter is locked.
    test_labels = joblib.load(args.test_parsed)
    test_filter_names = ("raw",) if selected_name == "raw" else ("raw", selected_name)
    test, test_rows = evaluate_split(
        "test",
        test_labels,
        test_indices(test_labels, image_root),
        0,
        test_filter_names,
        image_root,
        network,
        adapter,
        pose_model,
        hmr2s,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        args.pose_batch_size,
        args.hmr_batch_size,
        args.smpl_batch_size,
    )
    test_raw = test["variants"]["raw"]
    test_selected = test["variants"][selected_name]
    test_delta = {
        metric: float(test_selected["metrics"][metric]["mean"])
        - float(test_raw["metrics"][metric]["mean"])
        for metric in METRICS
    }
    test_relative = {
        metric: float(test_selected["metrics"][metric]["mean"])
        / max(float(test_raw["metrics"][metric]["mean"]), 1e-8)
        - 1.0
        for metric in METRICS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "causal_output_smoothing_ablation",
        "protocol": {
            "pipeline": "YOLO26m-pose + released HMR2.0-S + selected token adapter + released WHAM",
            "filter_location": "after WHAM pose/shape output; model inputs are unchanged",
            "filter_type": "causal geodesic exponential smoothing for rotations; causal EMA for shape",
            "selection_split": "3DPW validation",
            "test_split_opened_after_filter_lock": True,
            "selection_score": "40% PA-MPJPE, 25% MPJPE, 25% PVE, 10% acceleration, normalized to raw",
            "selection_gate": "no spatial validation metric may regress by more than 5%",
            "camera_signal": "zero angular velocity because 3DPW has no phone gyroscope stream",
            "latency_note": "group-delay estimates describe the pose EMA at low frequency; actual Core ML/Swift cost must be measured on iPhone",
        },
        "filters": {
            name: {
                **parameters,
                "approximate_low_frequency_group_delay_frames": (
                    0.0
                    if parameters["pose_alpha"] >= 1.0
                    else (1.0 - parameters["pose_alpha"]) / parameters["pose_alpha"]
                ),
                "approximate_low_frequency_group_delay_ms_at_30fps": (
                    0.0
                    if parameters["pose_alpha"] >= 1.0
                    else (1.0 - parameters["pose_alpha"])
                    / parameters["pose_alpha"]
                    * 1000.0
                    / 30.0
                ),
            }
            for name, parameters in FILTERS.items()
        },
        "validation": validation,
        "locked_filter": selected_name,
        "test": test,
        "decision": {
            "worth_smoothing": selected_name != "raw",
            "locked_filter": selected_name,
            "test_selected_minus_raw": test_delta,
            "test_selected_relative_change_vs_raw": test_relative,
            "raw_test_metrics": mean_metrics(test_raw),
            "selected_test_metrics": mean_metrics(test_selected),
        },
        "provenance": {
            "wham_commit": wham_commit,
            "hmr2s_commit": hmr2s_commit,
            "wham_checkpoint_sha256": sha256(args.wham_checkpoint),
            "hmr2s_checkpoint_sha256": sha256(args.hmr2s_checkpoint),
            "yolo26m_checkpoint_sha256": sha256(args.yolo26m_weights),
            "adapter_checkpoint_sha256": sha256(args.adapter_checkpoint),
            "val_parsed_sha256": sha256(args.val_parsed),
            "test_parsed_sha256": sha256(args.test_parsed),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    combined_rows = rows + test_rows
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in combined_rows for key in row})
        )
        writer.writeheader()
        writer.writerows(combined_rows)
    print(json.dumps(report["decision"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
