#!/usr/bin/env python3
"""Global BEDLAM adaptation with HMR2 regularization and 3DPW replay.

All accepted BEDLAM samples are pooled before optimization.  Training then
accumulates across complete global epochs, while the immutable source
checkpoint remains the fallback.  Candidate checkpoints are selected with
actual SMPL metrics on 3DPW validation; rejected epochs do not replace the
fallback, but they no longer erase learning before the next global epoch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import tarfile
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import distill_fastvit_hmr2 as distill
import evaluate_full_pipeline_tradeoff as full_eval
import evaluate_wham_feature_substitution as feature_eval
import finetune_fastvit_wham_downstream as phase3
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import train_bedlam_tiny_pipeline as bedlam
import train_deployment_tiny_pipeline as deployment
from torch import nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

SOURCE_DEPLOYMENT_SHA256 = (
    "d47ac0c855f44f84b67b20b9abb2cd3a66af702de479692350723cd79815c2e4"
)
HMR2_SHA256 = "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"
OUTPUT_CHECKPOINT = "bedlam_global_replay_best.pth"
PREVIOUS_TINY_TEST = {
    "pa_mpjpe_mm": 61.75674481248633,
    "mpjpe_mm": 109.36314765542959,
    "pve_mm": 127.2650627371855,
    "accel_official_30fps": 13.749177522877579,
}
METRIC_WEIGHTS = {
    "pa_mpjpe_mm": 0.40,
    "mpjpe_mm": 0.25,
    "pve_mm": 0.25,
    "accel_official_30fps": 0.10,
}
BEDLAM_DOWNSTREAM_WEIGHTS = {
    # Geometry is the primary BEDLAM objective. HMR2 matching is a regularizer.
    "token": 0.20,
    "cosine": 0.05,
    "init_pose": 1.00,
    "init_root": 2.00,
    "wham_pose": 2.00,
    "wham_root": 4.00,
    "wham_shape": 0.25,
}


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"per_track", "raw_values"}
    }


class Replay:
    def __init__(self, loader: DataLoader[dict[str, torch.Tensor]]) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


@torch.inference_mode()
def cache_hmr2_tokens(
    teacher: nn.Module,
    samples: list[bedlam.BedlamSample],
    batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    output = np.empty((len(samples), 1024), dtype=np.float16)
    started = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = bedlam.normalized_images(
            np.stack([sample.crop for sample in batch]), device
        )
        if device.type == "cuda":
            images = images.half()
        output[start : start + len(batch)] = (
            teacher(images).float().cpu().numpy().astype(np.float16)
        )
        if (start // batch_size + 1) % 25 == 0 or start + len(batch) == len(samples):
            print(
                json.dumps(
                    {
                        "hmr2_cache": label,
                        "samples": start + len(batch),
                        "total": len(samples),
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    return output


def student_token(
    student: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_map = student.backbone.forward_features(images[:, :, :, 32:-32])
    normalized = student.spatial_head(feature_map)
    return normalized, normalized * student.target_std + student.target_mean


def normalized_token_losses(
    student: nn.Module, prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    target_z = (
        target.float() - student.target_mean.float()
    ) / student.target_std.float()
    prediction_z = (
        prediction.float() - student.target_mean.float()
    ) / student.target_std.float()
    return (
        F.mse_loss(prediction_z, target_z),
        (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean(),
    )


def configure_token_stage(
    student: nn.Module, initializer: nn.Module, network: nn.Module
) -> list[dict[str, Any]]:
    for model in (student, initializer, network):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for parameter in student.spatial_head.parameters():
        parameter.requires_grad_(True)
    backbone_parameters: list[nn.Parameter] = []
    for module in (student.backbone.stages[3], student.backbone.final_conv):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            backbone_parameters.append(parameter)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return [
        {"params": list(student.spatial_head.parameters()), "lr": 8e-6},
        {"params": backbone_parameters, "lr": 5e-7},
    ]


def configure_mixed_stage(
    student: nn.Module, initializer: nn.Module, network: nn.Module
) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    for model in (student, initializer, network):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for parameter in initializer.parameters():
        parameter.requires_grad_(True)
    for parameter in student.spatial_head.parameters():
        parameter.requires_grad_(True)
    adapters: list[nn.Parameter] = [network.mask_embedding]
    network.mask_embedding.requires_grad_(True)
    for module in (
        network.motion_encoder.embed_layer,
        network.motion_encoder.neural_init,
        network.integrator,
        network.motion_decoder.neural_init,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            adapters.append(parameter)
    backbone_parameters: list[nn.Parameter] = []
    for module in (student.backbone.stages[3], student.backbone.final_conv):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            backbone_parameters.append(parameter)
    groups = [
        {"params": list(initializer.parameters()), "lr": 3e-5},
        {"params": list(student.spatial_head.parameters()), "lr": 5e-6},
        {"params": adapters, "lr": 2e-6},
        {"params": backbone_parameters, "lr": 2e-7},
    ]
    trainable = [
        parameter
        for model in (student, initializer, network)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return groups, trainable


def set_token_modes(
    student: nn.Module, initializer: nn.Module, network: nn.Module
) -> None:
    # The frozen recurrent modules still need cuDNN's training reserve buffer
    # because gradients flow through them into the FastViT token.
    deployment.set_modes(student, initializer, network, "last_stage")


def set_mixed_modes(
    student: nn.Module, initializer: nn.Module, network: nn.Module
) -> None:
    deployment.set_modes(student, initializer, network, "last_stage")


def token_replay_loss(
    student: nn.Module,
    samples: list[bedlam.BedlamSample],
    targets: np.ndarray,
    real_batch: dict[str, torch.Tensor],
    initializer: nn.Module,
    network: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = bedlam.normalized_images(
        np.stack([sample.crop for sample in samples]), device
    )
    _, bed_prediction = student_token(student, images)
    bed_target = torch.from_numpy(targets.astype(np.float32)).to(device)
    bed_mse, bed_cosine = normalized_token_losses(student, bed_prediction, bed_target)
    real_batch = deployment.move_batch(real_batch, device)
    real_loss, real_components = deployment.compute_losses(
        student, initializer, network, real_batch, "last_stage"
    )
    total = 0.50 * (bed_mse + 0.25 * bed_cosine) + real_loss
    return total, {
        "bed_token": bed_mse,
        "bed_cosine": bed_cosine,
        "real_loss": real_loss,
        "real_token": real_components["token"],
        "real_pose": real_components["wham_pose"],
    }


def clip_target_tokens(
    clips: list[list[bedlam.BedlamSample]],
    token_by_sample: dict[int, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [[token_by_sample[id(sample)] for sample in clip] for clip in clips]
    ).astype(np.float32)


def synthetic_clip_loss(
    student: nn.Module,
    initializer: nn.Module,
    network: nn.Module,
    clips: list[list[bedlam.BedlamSample]],
    teacher_tokens: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    crop_array = np.stack([[sample.crop for sample in clip] for clip in clips])
    images = bedlam.normalized_images(crop_array, device)
    batch_size, sequence_length = images.shape[:2]
    flat = images.reshape(-1, 3, 256, 256)
    normalized, token = student_token(student, flat)
    del normalized
    token = token.reshape(batch_size, sequence_length, 1024)
    target_token = torch.from_numpy(teacher_tokens).to(device)
    token_mse, cosine = normalized_token_losses(student, token, target_token)
    initial_pose, initial_joints = initializer(token)
    target_axis = torch.from_numpy(
        np.stack([[sample.pose for sample in clip] for clip in clips])
    ).to(device)
    target_pose = feature_eval.matrix_to_rotation_6d(
        feature_eval.axis_angle_to_matrix(target_axis)
    )
    target_shape = torch.from_numpy(
        np.stack([[sample.shape for sample in clip] for clip in clips])
    ).to(device)
    x = torch.from_numpy(
        np.stack([[sample.x for sample in clip] for clip in clips])
    ).to(device)
    mask = torch.from_numpy(
        np.stack([[sample.mask for sample in clip] for clip in clips])
    ).to(device)
    valid = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    predicted_pose, _, predicted_shape = deployment.pose_core(
        network,
        x,
        mask,
        token,
        valid,
        initial_joints[:, 0],
        initial_pose,
    )
    init_error = deployment.rotation_loss(initial_pose, target_pose)
    wham_error = deployment.rotation_loss(predicted_pose, target_pose[:, 1:])
    components = {
        "token": token_mse,
        "cosine": cosine,
        "init_pose": init_error.mean(),
        "init_root": init_error[..., 0].mean(),
        "wham_pose": wham_error.mean(),
        "wham_root": wham_error[..., 0].mean(),
        "wham_shape": F.smooth_l1_loss(
            predicted_shape.float(), target_shape[:, 1:].float(), beta=0.5
        ),
    }
    total = sum(
        BEDLAM_DOWNSTREAM_WEIGHTS[name] * value for name, value in components.items()
    )
    return total, components


def optimize_token_stage(
    student: nn.Module,
    initializer: nn.Module,
    network: nn.Module,
    samples: list[bedlam.BedlamSample],
    teacher_tokens: np.ndarray,
    replay: Replay,
    device: torch.device,
    batch_size: int,
    maximum_steps: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    trainable: list[nn.Parameter],
) -> dict[str, float]:
    order = list(range(len(samples)))
    random.shuffle(order)
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    started = time.perf_counter()
    set_token_modes(student, initializer, network)
    for indices in chunks(order, batch_size):
        if steps >= maximum_steps:
            break
        batch_samples = [samples[index] for index in indices]
        batch_targets = teacher_tokens[np.asarray(indices, dtype=np.int64)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss, components = token_replay_loss(
                student,
                batch_samples,
                batch_targets,
                replay.next(),
                initializer,
                network,
                device,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        totals["loss"] += float(loss.detach())
        for key, value in components.items():
            totals[key] += float(value.detach())
        if steps % 25 == 0:
            print(
                json.dumps(
                    {
                        "training_stage": "global_hmr2_warmup",
                        "step": steps,
                        "maximum_steps": maximum_steps,
                        "loss": totals["loss"] / steps,
                        "bed_cosine_loss": totals["bed_cosine"] / steps,
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    return {
        "steps": steps,
        **{key: value / max(steps, 1) for key, value in totals.items()},
    }


def optimize_mixed_stage(
    student: nn.Module,
    initializer: nn.Module,
    network: nn.Module,
    clips: list[list[bedlam.BedlamSample]],
    token_by_sample: dict[int, np.ndarray],
    replay: Replay,
    device: torch.device,
    batch_size: int,
    maximum_steps: int,
    bedlam_weight: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    trainable: list[nn.Parameter],
) -> dict[str, float]:
    random.shuffle(clips)
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    started = time.perf_counter()
    set_mixed_modes(student, initializer, network)
    for batch_clips in chunks(clips, batch_size):
        if steps >= maximum_steps:
            break
        target_tokens = clip_target_tokens(batch_clips, token_by_sample)
        real_batch = deployment.move_batch(replay.next(), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            synthetic_loss, synthetic_components = synthetic_clip_loss(
                student,
                initializer,
                network,
                batch_clips,
                target_tokens,
                device,
            )
            real_loss, real_components = deployment.compute_losses(
                student, initializer, network, real_batch, "last_stage"
            )
            loss = real_loss + bedlam_weight * synthetic_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        totals["loss"] += float(loss.detach())
        totals["real_loss"] += float(real_loss.detach())
        totals["bedlam_loss"] += float(synthetic_loss.detach())
        totals["real_pose"] += float(real_components["wham_pose"].detach())
        totals["bedlam_pose"] += float(synthetic_components["wham_pose"].detach())
        totals["bedlam_token"] += float(synthetic_components["token"].detach())
        if steps % 25 == 0:
            print(
                json.dumps(
                    {
                        "training_stage": "global_geometry_replay",
                        "step": steps,
                        "maximum_steps": maximum_steps,
                        "loss": totals["loss"] / steps,
                        "bedlam_pose": totals["bedlam_pose"] / steps,
                        "real_pose": totals["real_pose"] / steps,
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    return {
        "steps": steps,
        **{key: value / max(steps, 1) for key, value in totals.items()},
    }


@torch.inference_mode()
def metric_validation(
    student: nn.Module,
    initializer: nn.Module,
    network: nn.Module,
    labels: dict[str, Any],
    cached_tracks: list[dict[str, Any]],
    smpl_models: dict[str, nn.Module],
    joint_regressor: torch.Tensor,
    feature_batch_size: int,
    smpl_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    student.eval()
    initializer.eval()
    network.eval()
    accumulated: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    per_track: list[dict[str, Any]] = []
    for track in cached_tracks:
        tokens: list[torch.Tensor] = []
        for start in range(0, len(track["paths"]), feature_batch_size):
            stop = min(start + feature_batch_size, len(track["paths"]))
            images = deployment.observation_images(
                track["paths"][start:stop],
                track["crop_box"][start:stop],
                track["valid"][start:stop],
            ).to(device, non_blocking=True)
            output = student(images)
            output = output * torch.from_numpy(track["valid"][start:stop]).to(
                device
            ).unsqueeze(-1)
            tokens.append(output.float())
        token = torch.cat(tokens).unsqueeze(0)
        x = torch.from_numpy(track["x"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(track["mask"]).unsqueeze(0).to(device)
        initial_pose, initial_joints = initializer(token[:, :1])
        output = deployment.deployment_core(
            network=network,
            x=x[:, 1:],
            mask=mask[:, 1:],
            features=token[:, 1:],
            feature_valid=torch.from_numpy(track["valid"][1:]).unsqueeze(0).to(device),
            init_kp=torch.cat((initial_joints.reshape(1, 1, 51), x[:, :1]), dim=-1),
            init_pose=initial_pose,
            init_root=initial_pose[:, :, 0],
            cam_angvel=torch.zeros(1, track["available"], 6, device=device),
        )
        index = int(track["index"])
        available = int(track["available"])
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
        values, _ = full_eval.smpl_metrics(
            output,
            target_pose,
            target_betas,
            gender,
            smpl_models,
            joint_regressor,
            device,
            smpl_batch_size,
        )
        row = {"sequence": track["video_id"], "frames": available}
        for name, metric_values in values.items():
            accumulated[name].append(metric_values)
            row[name] = float(metric_values.mean())
        per_track.append(row)
    result = {
        name: float(np.concatenate(values).mean())
        for name, values in accumulated.items()
    }
    result["tracks"] = len(per_track)
    result["frames"] = int(sum(row["frames"] for row in per_track))
    result["per_track"] = per_track
    return result


def normalized_metric_score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    return float(
        sum(
            METRIC_WEIGHTS[name] * float(metrics[name]) / float(baseline[name])
            for name in METRIC_WEIGHTS
        )
    )


def track_wins(candidate: dict[str, Any], incumbent: dict[str, Any]) -> int:
    incumbent_by_name = {row["sequence"]: row for row in incumbent["per_track"]}
    return sum(
        row["pa_mpjpe_mm"] < incumbent_by_name[row["sequence"]]["pa_mpjpe_mm"]
        for row in candidate["per_track"]
    )


def candidate_accepted(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    candidate_score = normalized_metric_score(candidate, baseline)
    incumbent_score = normalized_metric_score(incumbent, baseline)
    wins = track_wins(candidate, incumbent)
    required_wins = math.ceil(0.50 * len(candidate["per_track"]))
    gates = {
        "composite_improves_by_0_3_percent": candidate_score <= incumbent_score - 0.003,
        "pa_mpjpe_improves": candidate["pa_mpjpe_mm"] < incumbent["pa_mpjpe_mm"],
        "mpjpe_within_1_percent_of_baseline": candidate["mpjpe_mm"]
        <= 1.01 * baseline["mpjpe_mm"],
        "pve_within_1_percent_of_baseline": candidate["pve_mm"]
        <= 1.01 * baseline["pve_mm"],
        "accel_within_5_percent_of_baseline": candidate["accel_official_30fps"]
        <= 1.05 * baseline["accel_official_30fps"],
        "wins_at_least_half_of_tracks": wins >= required_wins,
    }
    return all(gates.values()), {
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "pa_track_wins": wins,
        "required_pa_track_wins": required_wins,
        "gates": gates,
    }


def save_checkpoint(
    source: dict[str, Any],
    student: nn.Module,
    initializer: nn.Module,
    network: nn.Module,
    path: Path,
    epoch: int,
    stage: str,
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    payload = deployment.checkpoint_payload(
        source,
        student,
        initializer,
        network,
        epoch,
        stage,
        compact_metrics(validation),
        metadata,
    )
    torch.save(payload, path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bedlam-label-root", type=Path)
    parser.add_argument("--hf-repo", default=bedlam.HF_REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--three-dpw-root", type=Path)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--train-parsed", type=Path)
    parser.add_argument("--val-parsed", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--wham-repo", type=Path)
    parser.add_argument("--wham-checkpoint", type=Path)
    parser.add_argument("--yolo26-weights", type=Path)
    parser.add_argument("--hmr2-checkpoint", type=Path)
    parser.add_argument("--smpl-model-directory", type=Path)
    parser.add_argument("--h36m-joint-regressor", type=Path)
    parser.add_argument("--maximum-scenes", type=int, default=12)
    parser.add_argument("--maximum-download-gib", type=float, default=4.5)
    parser.add_argument("--videos-per-scene", type=int, default=24)
    parser.add_argument("--frames-per-video", type=int, default=30)
    parser.add_argument("--clips-per-video", type=int, default=12)
    parser.add_argument("--bedlam-clip-length", type=int, default=8)
    parser.add_argument("--bedlam-clip-stride", type=int, default=4)
    parser.add_argument("--bedlam-frame-batch-size", type=int, default=20)
    parser.add_argument("--bedlam-clip-batch-size", type=int, default=2)
    parser.add_argument("--hmr2-batch-size", type=int, default=6)
    parser.add_argument("--token-epochs", type=int, default=2)
    parser.add_argument("--mixed-epochs", type=int, default=8)
    parser.add_argument("--token-steps-per-epoch", type=int, default=256)
    parser.add_argument("--mixed-steps-per-epoch", type=int, default=256)
    parser.add_argument("--bedlam-weight", type=float, default=0.30)
    parser.add_argument("--minimum-bedlam-samples", type=int, default=200)
    parser.add_argument("--minimum-global-samples", type=int, default=3000)
    parser.add_argument("--minimum-global-clips", type=int, default=150)
    parser.add_argument("--divergence-score", type=float, default=1.03)
    parser.add_argument("--yolo-batch-size", type=int, default=32)
    parser.add_argument("--three-dpw-clip-length", type=int, default=24)
    parser.add_argument("--three-dpw-stride", type=int, default=12)
    parser.add_argument("--three-dpw-max-clips", type=int, default=1600)
    parser.add_argument("--three-dpw-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--val-tracks", type=int, default=16)
    parser.add_argument("--val-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-8
    assert (
        bedlam.expected_remote_candidates(Path("scene_6fps.npz"))[0]
        == "scene/mp4/scene_mp4.tar"
    )
    baseline = {
        **{name: 100.0 for name in METRIC_WEIGHTS},
        "per_track": [{"sequence": f"s{i}", "pa_mpjpe_mm": 100.0} for i in range(4)],
    }
    improved = copy.deepcopy(baseline)
    for name in METRIC_WEIGHTS:
        improved[name] = 99.0
    for row in improved["per_track"]:
        row["pa_mpjpe_mm"] = 99.0
    accepted, diagnostics = candidate_accepted(improved, baseline, baseline)
    assert accepted and all(diagnostics["gates"].values())
    print("Self-test passed: official BEDLAM path and metric-aligned acceptance")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.bedlam_label_root,
        args.output_dir,
        args.scratch_dir,
        args.cache_dir,
        args.three_dpw_root,
        args.train_parsed,
        args.val_parsed,
        args.source_checkpoint,
        args.wham_repo,
        args.wham_checkpoint,
        args.yolo26_weights,
        args.hmr2_checkpoint,
        args.smpl_model_directory,
        args.h36m_joint_regressor,
    )
    if any(path is None for path in required):
        raise ValueError("All paths are required outside --self-test")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing")
    if feature_eval.sha256_file(args.source_checkpoint) != SOURCE_DEPLOYMENT_SHA256:
        raise RuntimeError(
            "The source must be the previous winning deployment checkpoint"
        )
    if feature_eval.sha256_file(args.hmr2_checkpoint) != HMR2_SHA256:
        raise RuntimeError("HMR2 checkpoint checksum mismatch")
    if feature_eval.sha256_file(args.yolo26_weights) != deployment.YOLO26_SHA256:
        raise RuntimeError("YOLO26 checkpoint checksum mismatch")
    label_paths = sorted(
        path
        for path in args.bedlam_label_root.glob("*.npz")
        if path.name != "agora.npz"
    )
    if len(label_paths) < 20:
        raise RuntimeError("The BEDLAM label package is incomplete")
    scenes, unmatched = bedlam.resolve_remote_scenes(label_paths, args.hf_repo, token)
    selected_scenes = bedlam.select_scenes(
        scenes, args.maximum_scenes, args.maximum_download_gib
    )
    preflight = {
        "labels": len(label_paths),
        "matched_archives": len(scenes),
        "unmatched": unmatched,
        "selected": [scene.remote_path for scene in selected_scenes],
        "selected_download_gib": sum(scene.remote_size for scene in selected_scenes)
        / (1024**3),
        "source_sha256": SOURCE_DEPLOYMENT_SHA256,
        "hmr2_sha256": HMR2_SHA256,
        "selection": "actual 3DPW validation PA-MPJPE/MPJPE/PVE/acceleration",
    }
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.inspect_data:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    image_root = feature_eval.locate_image_root(args.three_dpw_root)
    sequence_root = phase3.locate_sequence_train_root(
        args.sequence_root or args.three_dpw_root
    )
    train_tracks = phase3.map_training_tracks(train_labels, sequence_root)
    val_indices = deployment.validation_indices(val_labels, image_root, args.val_tracks)

    print("Caching real YOLO26 observations for replay and validation...", flush=True)
    pose_model = YOLO(str(args.yolo26_weights))
    train_paths, train_observations = deployment.cache_training_observations(
        args.cache_dir / "replay_train_yolo26.pth",
        train_labels,
        train_tracks,
        image_root,
        pose_model,
        args.yolo_batch_size,
        device,
        {
            "schema": 3,
            "split": "3dpw_train_replay",
            "parsed_sha256": feature_eval.sha256_file(args.train_parsed),
            "yolo_sha256": deployment.YOLO26_SHA256,
            "confidence_threshold": 0.5,
            "minimum_iou": 0.2,
        },
    )
    val_cache = deployment.cache_validation_observations(
        args.cache_dir / "metric_val_yolo26.pth",
        val_labels,
        val_indices,
        image_root,
        pose_model,
        args.yolo_batch_size,
        args.val_frames,
        device,
        {
            "schema": 3,
            "split": "3dpw_validation_metric_selection",
            "parsed_sha256": feature_eval.sha256_file(args.val_parsed),
            "yolo_sha256": deployment.YOLO26_SHA256,
            "indices": val_indices,
            "max_frames": args.val_frames,
        },
    )
    replay_dataset = deployment.DeploymentClipDataset(
        train_labels,
        train_tracks,
        train_paths,
        train_observations,
        args.three_dpw_clip_length,
        args.three_dpw_stride,
        args.three_dpw_max_clips,
        minimum_valid_rate=0.75,
        seed=args.seed,
    )
    replay_loader = DataLoader(
        replay_dataset,
        batch_size=args.three_dpw_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    replay = Replay(replay_loader)

    student, initializer, network, source = __import__(
        "evaluate_deployment_tiny_pipeline_3dpw"
    ).load_deployment_models(
        args.source_checkpoint, args.wham_repo, args.wham_checkpoint, device
    )
    smpl_models = full_eval.load_smpl_models(args.smpl_model_directory, device)
    h36m_np = np.load(args.h36m_joint_regressor)[full_eval.H36M_TO_J14, :]
    joint_regressor = torch.from_numpy(h36m_np).float().unsqueeze(0).to(device)
    teacher, _ = distill.load_teacher(args.wham_repo, args.hmr2_checkpoint, device)
    metadata = dict(source.get("deployment_metadata") or {})
    metadata.update(
        {
            "bedlam_global_replay_schema": 2,
            "bedlam_teacher": "official_HMR2_token_encoder",
            "real_replay_every_synthetic_update": True,
            "global_pool_training": True,
            "global_epoch_selection": True,
            "geometry_is_primary_bedlam_objective": True,
            "fastvit_last_stage_unfrozen": True,
            "bedlam_detector_minimum_iou": 0.45,
            "bedlam_detector_confidence": 0.35,
            "metric_aligned_validation": True,
            "test_data_used": False,
        }
    )

    baseline = metric_validation(
        student,
        initializer,
        network,
        val_labels,
        val_cache,
        smpl_models,
        joint_regressor,
        args.feature_batch_size,
        args.smpl_batch_size,
        device,
    )
    best_validation = baseline
    best_stage = "source_deployment_baseline"
    best_epoch = 0
    best_path = args.output_dir / OUTPUT_CHECKPOINT
    save_checkpoint(
        source,
        student,
        initializer,
        network,
        best_path,
        0,
        best_stage,
        baseline,
        metadata,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "stage": best_stage,
            "accepted": True,
            "score": 1.0,
            **compact_metrics(baseline),
        }
    ]
    manifest: list[dict[str, Any]] = []
    print(
        json.dumps({"baseline_metric_validation": compact_metrics(baseline)}, indent=2),
        flush=True,
    )

    # Build one global pool first. Nothing is optimized scene-by-scene.
    global_samples: list[bedlam.BedlamSample] = []
    global_token_chunks: list[np.ndarray] = []
    global_clips: list[list[bedlam.BedlamSample]] = []
    for scene_ordinal, scene in enumerate(selected_scenes, start=1):
        scene_started = time.perf_counter()
        scene_cache = args.scratch_dir / f"hf_scene_{scene_ordinal:02d}"
        shutil.rmtree(scene_cache, ignore_errors=True)
        print(f"Downloading {scene.remote_path}", flush=True)
        archive_path = Path(
            __import__("huggingface_hub").hf_hub_download(
                repo_id=args.hf_repo,
                filename=scene.remote_path,
                repo_type="dataset",
                revision="main",
                token=token,
                cache_dir=scene_cache,
            )
        )
        scene_samples: list[bedlam.BedlamSample] = []
        scene_tokens: list[np.ndarray] = []
        attempted = 0
        with (
            np.load(scene.label_path, allow_pickle=False) as labels,
            tarfile.open(archive_path, "r:*") as archive,
        ):
            scene_labels = {
                key: labels[key]
                for key in ("imgname", "center", "scale", "pose_cam", "shape", "gender")
            }
            sequence_rows = bedlam.map_sequence_rows(scene_labels["imgname"])
            members = bedlam.tar_video_members(archive)
            available = sorted(
                sequence
                for sequence in sequence_rows
                if bedlam.locate_tar_member(members, sequence) is not None
            )
            chosen = bedlam.evenly_spaced(available, args.videos_per_scene)
            for video_ordinal, sequence in enumerate(chosen, start=1):
                member = bedlam.locate_tar_member(members, sequence)
                if member is None:
                    continue
                video_path = args.scratch_dir / "active_bedlam_video.mp4"
                bedlam.extract_member(archive, member, video_path)
                try:
                    samples, stats = bedlam.samples_from_video(
                        scene_labels,
                        sequence_rows[sequence],
                        video_path,
                        pose_model,
                        device,
                        args.frames_per_video,
                        args.yolo_batch_size,
                        rotate_clockwise="closeup" in scene.label_scene.lower(),
                        confidence_threshold=0.35,
                        minimum_iou=0.45,
                    )
                finally:
                    video_path.unlink(missing_ok=True)
                attempted += int(stats["attempted_person_frames"])
                if samples:
                    tokens = cache_hmr2_tokens(
                        teacher,
                        samples,
                        args.hmr2_batch_size,
                        device,
                        f"{scene_ordinal}/{len(selected_scenes)}:{video_ordinal}/{len(chosen)}",
                    )
                    scene_samples.extend(samples)
                    scene_tokens.append(tokens)
        shutil.rmtree(scene_cache, ignore_errors=True)
        accepted_samples = len(scene_samples)
        if accepted_samples < args.minimum_bedlam_samples:
            row = {
                "epoch": 0,
                "stage": "scene_skipped_low_match_count",
                "scene": scene.label_scene,
                "attempted": attempted,
                "accepted_samples": accepted_samples,
                "detection_rate": accepted_samples / max(attempted, 1),
                "accepted": False,
            }
            history.append(row)
            manifest.append(row)
            write_history(args.output_dir / "bedlam_global_replay_history.csv", history)
            (args.output_dir / "bedlam_global_replay_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row, indent=2), flush=True)
            continue
        teacher_tokens = np.concatenate(scene_tokens)
        clips = bedlam.build_temporal_clips(
            scene_samples,
            args.bedlam_clip_length,
            args.bedlam_clip_stride,
            args.clips_per_video * max(len(chosen), 1),
        )
        scene_row = {
            "scene": scene.label_scene,
            "remote": scene.remote_path,
            "attempted": attempted,
            "accepted_samples": accepted_samples,
            "detection_rate": accepted_samples / max(attempted, 1),
            "clips": len(clips),
            "minutes": (time.perf_counter() - scene_started) / 60.0,
        }
        manifest.append(scene_row)
        global_samples.extend(scene_samples)
        global_token_chunks.append(teacher_tokens)
        global_clips.extend(clips)
        (args.output_dir / "bedlam_global_replay_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    **scene_row,
                    "global_samples": len(global_samples),
                    "global_clips": len(global_clips),
                },
                indent=2,
            ),
            flush=True,
        )

    if len(global_samples) < args.minimum_global_samples:
        raise RuntimeError(
            f"Global BEDLAM pool has only {len(global_samples)} accepted samples; "
            f"need at least {args.minimum_global_samples}"
        )
    if len(global_clips) < args.minimum_global_clips:
        raise RuntimeError(
            f"Global BEDLAM pool has only {len(global_clips)} clips; "
            f"need at least {args.minimum_global_clips}"
        )
    teacher_tokens = np.concatenate(global_token_chunks)
    if len(teacher_tokens) != len(global_samples):
        raise RuntimeError("Global BEDLAM samples and HMR2 tokens are misaligned")
    token_by_sample = {
        id(sample): teacher_tokens[index]
        for index, sample in enumerate(global_samples)
    }
    pool_summary = {
        "samples": len(global_samples),
        "clips": len(global_clips),
        "scenes": sum("remote" in row for row in manifest),
        "attempted": sum(int(row.get("attempted", 0)) for row in manifest),
    }
    print(json.dumps({"global_pool": pool_summary}, indent=2), flush=True)

    # HMR2 and YOLO are no longer needed in GPU memory after the global pool exists.
    del teacher, pose_model, global_token_chunks
    torch.cuda.empty_cache()

    epoch = 0
    history_path = args.output_dir / "bedlam_global_replay_history.csv"

    def evaluate_global_epoch(stage: str, training: dict[str, float]) -> bool:
        nonlocal best_validation, best_stage, best_epoch
        candidate = metric_validation(
            student,
            initializer,
            network,
            val_labels,
            val_cache,
            smpl_models,
            joint_regressor,
            args.feature_batch_size,
            args.smpl_batch_size,
            device,
        )
        eligible, selection = candidate_accepted(candidate, baseline, baseline)
        candidate_score = normalized_metric_score(candidate, baseline)
        best_score = normalized_metric_score(best_validation, baseline)
        selected = eligible and candidate_score < best_score
        row = {
            "epoch": epoch,
            "stage": stage,
            "accepted": selected,
            "eligible_against_source": eligible,
            "improves_selected_checkpoint": candidate_score < best_score,
            **training,
            **{
                f"val_{key}": value
                for key, value in compact_metrics(candidate).items()
            },
            **selection,
        }
        history.append(row)
        if selected:
            best_validation = candidate
            best_stage = stage
            best_epoch = epoch
            save_checkpoint(
                source,
                student,
                initializer,
                network,
                best_path,
                epoch,
                best_stage,
                candidate,
                metadata,
            )
        write_history(history_path, history)
        print(json.dumps(row, indent=2), flush=True)
        finite = all(
            math.isfinite(float(candidate[name])) for name in METRIC_WEIGHTS
        )
        return finite and candidate_score <= args.divergence_score

    # Phase A: global feature-manifold warmup. Adam state persists across epochs.
    token_groups = configure_token_stage(student, initializer, network)
    token_trainable = [
        parameter for group in token_groups for parameter in group["params"]
    ]
    token_optimizer = torch.optim.AdamW(token_groups, weight_decay=0.01)
    token_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for _ in range(args.token_epochs):
        epoch += 1
        token_training = optimize_token_stage(
            student,
            initializer,
            network,
            global_samples,
            teacher_tokens,
            replay,
            device,
            args.bedlam_frame_batch_size,
            args.token_steps_per_epoch,
            token_optimizer,
            token_scaler,
            token_trainable,
        )
        if not evaluate_global_epoch("global_hmr2_warmup", token_training):
            print("Warmup diverged; restoring the selected fallback before geometry training")
            del token_optimizer, token_scaler, token_trainable, token_groups
            del student, initializer, network
            torch.cuda.empty_cache()
            student, initializer, network, _ = __import__(
                "evaluate_deployment_tiny_pipeline_3dpw"
            ).load_deployment_models(
                best_path, args.wham_repo, args.wham_checkpoint, device
            )
            break
    else:
        del token_optimizer, token_scaler, token_trainable, token_groups

    # Phase B: geometry-first training. Adam state again persists across epochs.
    mixed_groups, mixed_trainable = configure_mixed_stage(
        student, initializer, network
    )
    mixed_optimizer = torch.optim.AdamW(mixed_groups, weight_decay=0.01)
    mixed_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for _ in range(args.mixed_epochs):
        epoch += 1
        mixed_training = optimize_mixed_stage(
            student,
            initializer,
            network,
            global_clips,
            token_by_sample,
            replay,
            device,
            args.bedlam_clip_batch_size,
            args.mixed_steps_per_epoch,
            args.bedlam_weight,
            mixed_optimizer,
            mixed_scaler,
            mixed_trainable,
        )
        if not evaluate_global_epoch("global_geometry_replay", mixed_training):
            print("Geometry training diverged; keeping the last selected checkpoint")
            break

    report = {
        "schema_version": 2,
        "experiment": (
            "global pooled BEDLAM geometry training with HMR2 regularization "
            "and one-for-one 3DPW replay"
        ),
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "baseline_validation": baseline,
        "best_validation": best_validation,
        "baseline_composite": normalized_metric_score(baseline, baseline),
        "best_composite": normalized_metric_score(best_validation, baseline),
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": feature_eval.sha256_file(best_path),
        "source_checkpoint_sha256": SOURCE_DEPLOYMENT_SHA256,
        "previous_tiny_test": PREVIOUS_TINY_TEST,
        "preflight": preflight,
        "global_pool": pool_summary,
        "manifest": manifest,
        "history_rows": len(history),
        "selection_policy": (
            "global-epoch actual 3DPW validation; select only >=0.3% composite "
            "improvement versus source, PA-MPJPE improvement, bounded regressions, "
            "and >=50% source-track wins; rejected epochs continue accumulating "
            "unless the composite exceeds the divergence guard"
        ),
        "test_data_used": False,
    }
    (args.output_dir / "bedlam_global_replay_training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_history(args.output_dir / "bedlam_global_replay_history.csv", history)
    print(
        json.dumps(
            {
                "best_stage": best_stage,
                "best_epoch": best_epoch,
                "baseline": compact_metrics(baseline),
                "best": compact_metrics(best_validation),
                "checkpoint": str(best_path),
                "checkpoint_sha256": report["best_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
