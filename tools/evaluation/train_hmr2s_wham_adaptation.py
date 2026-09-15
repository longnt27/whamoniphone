#!/usr/bin/env python3
"""Train and fairly evaluate two fixes for HMR2-S/WHAM token mismatch.

Candidate A learns an HMR2-S -> HMR2a token adapter while released WHAM is
frozen.  Candidate B starts independently from released WHAM and tunes its
image integrator/decoder for native HMR2-S tokens.  Candidate selection uses
3DPW validation.  The 3DPW test split is opened only after both checkpoints
have been locked, and is used once for a report-ready comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_frozen_hmr2s_wham as frozen_eval
import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as wham_eval
import finetune_fastvit_wham_downstream as phase3
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import train_bedlam_tiny_pipeline as bedlam
import train_deployment_tiny_pipeline as deployment
from distill_fastvit_hmr2 import (
    HMR2TokenEncoder,
    PersonCropDataset,
    PoseReadout,
    build_records,
    discover_split,
    load_teacher,
    validate_record_images,
)
from distill_fastvit_hmr2 import (
    rotation_6d_to_matrix as hmr2_rotation_6d_to_matrix,
)
from hmr2s_frozen import FrozenHMR2S, checkpoint_smpl_buffers
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

SCHEMA_VERSION = 1
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
PAIR_DOMAINS = ("coco2017_train", "3dpw_train", "bedlam_train")
FINAL_VARIANTS = (
    "released_wham",
    "naive_hmr2s_released_wham",
    "adapter_hmr2s_released_wham",
    "native_hmr2s_tuned_wham",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_log(**values: Any) -> None:
    print(json.dumps(values, sort_keys=True, default=str), flush=True)


def mean_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics[name]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


class TokenAdapter(nn.Module):
    """Small Core-ML-friendly residual map, initialized as the direct plug."""

    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(1024)
        self.in_projection = nn.Linear(1024, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, 1024)
        nn.init.xavier_uniform_(self.in_projection.weight)
        nn.init.zeros_(self.in_projection.bias)
        nn.init.zeros_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)
        self.register_buffer("target_mean", torch.zeros(1024))
        self.register_buffer("target_std", torch.ones(1024))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        correction = self.out_projection(F.gelu(self.in_projection(self.norm(token))))
        return token + correction


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        sources: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        domains: Sequence[str],
    ) -> None:
        if not (len(sources) == len(targets) == len(domains)):
            raise ValueError("PairDataset arguments have different lengths")
        source_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        domain_parts: list[np.ndarray] = []
        for domain_id, (source, target) in enumerate(zip(sources, targets)):
            if source.shape != target.shape or source.shape[-1] != 1024:
                raise ValueError(
                    f"Bad token pair shape: {source.shape}, {target.shape}"
                )
            source_parts.append(np.asarray(source, dtype=np.float32))
            target_parts.append(np.asarray(target, dtype=np.float32))
            domain_parts.append(np.full(len(source), domain_id, dtype=np.int64))
        self.source = np.concatenate(source_parts)
        self.target = np.concatenate(target_parts)
        self.domain = np.concatenate(domain_parts)
        self.domain_names = list(domains)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.from_numpy(self.source[index]),
            torch.from_numpy(self.target[index]),
            int(self.domain[index]),
        )


@dataclass(frozen=True)
class HMRCache:
    token: np.ndarray
    pose: np.ndarray
    betas: np.ndarray


def empty_hmr_cache(length: int) -> HMRCache:
    return HMRCache(
        token=np.zeros((length, 1024), dtype=np.float16),
        pose=np.zeros((length, 24, 6), dtype=np.float16),
        betas=np.zeros((length, 10), dtype=np.float16),
    )


def normalized_path_crops(
    paths: Sequence[Path | str],
    crop_boxes: np.ndarray,
    indices: Sequence[int],
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for index in indices:
        center_x, center_y, side = crop_boxes[index]
        with Image.open(paths[index]) as source:
            crop = mobile_eval.square_crop(
                source.convert("RGB"), float(center_x), float(center_y), float(side)
            )
        images.append(mobile_eval.image_to_tensor(crop))
    return torch.stack(images)


def path_crop_batches(
    paths: Sequence[Path | str],
    crop_boxes: np.ndarray,
    valid: np.ndarray,
    batch_size: int,
) -> Iterator[tuple[list[int], torch.Tensor]]:
    indices = np.flatnonzero(valid).astype(np.int64).tolist()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield batch_indices, normalized_path_crops(paths, crop_boxes, batch_indices)


def bedlam_crop_batches(
    samples: Sequence[bedlam.BedlamSample], batch_size: int
) -> Iterator[tuple[list[int], torch.Tensor]]:
    for start in range(0, len(samples), batch_size):
        indices = list(range(start, min(start + batch_size, len(samples))))
        array = np.stack([samples[index].crop for index in indices])
        yield indices, bedlam.normalized_images(array, torch.device("cpu"))


@torch.inference_mode()
def cache_teacher_from_batches(
    teacher: HMR2TokenEncoder,
    length: int,
    batches: Iterable[tuple[list[int], torch.Tensor]],
    device: torch.device,
    label: str,
) -> np.ndarray:
    output = np.zeros((length, 1024), dtype=np.float16)
    seen = 0
    for indices, images in tqdm(batches, desc=f"HMR2a {label}", leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.half()
        token = teacher(images).float().cpu().numpy().astype(np.float16)
        output[np.asarray(indices, dtype=np.int64)] = token
        seen += len(indices)
    json_log(cache=label, model="HMR2a", samples=seen)
    return output


@torch.inference_mode()
def cache_hmr2s_from_batches(
    model: FrozenHMR2S,
    length: int,
    batches: Iterable[tuple[list[int], torch.Tensor]],
    device: torch.device,
    label: str,
) -> HMRCache:
    cached = empty_hmr_cache(length)
    seen = 0
    for indices, images in tqdm(batches, desc=f"HMR2-S {label}", leave=False):
        images = images.to(device, non_blocking=True)
        token, pose, betas, _ = model(images)
        rows = np.asarray(indices, dtype=np.int64)
        cached.token[rows] = token.float().cpu().numpy().astype(np.float16)
        cached.pose[rows] = pose.float().cpu().numpy().astype(np.float16)
        cached.betas[rows] = betas.float().cpu().numpy().astype(np.float16)
        seen += len(indices)
    json_log(cache=label, model="HMR2-S", samples=seen)
    return cached


@torch.inference_mode()
def cache_coco_teacher(
    teacher: HMR2TokenEncoder,
    dataset: PersonCropDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    output = np.empty((len(dataset), 1024), dtype=np.float16)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    for images, indices in tqdm(loader, desc=f"HMR2a {label}", leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.half()
        token = teacher(images).float().cpu().numpy().astype(np.float16)
        output[indices.numpy()] = token
    return output


@torch.inference_mode()
def cache_coco_hmr2s(
    model: FrozenHMR2S,
    dataset: PersonCropDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    label: str,
) -> HMRCache:
    output = empty_hmr_cache(len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    for images, indices in tqdm(loader, desc=f"HMR2-S {label}", leave=False):
        images = images.to(device, non_blocking=True)
        token, pose, betas, _ = model(images)
        rows = indices.numpy()
        output.token[rows] = token.float().cpu().numpy().astype(np.float16)
        output.pose[rows] = pose.float().cpu().numpy().astype(np.float16)
        output.betas[rows] = betas.float().cpu().numpy().astype(np.float16)
    return output


def release_cuda(*objects: Any) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_evenly(values: list[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return [values[int(position)] for position in positions]


def cache_validation_hmr2s(
    model: FrozenHMR2S,
    tracks: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> None:
    for ordinal, track in enumerate(tracks, start=1):
        cached = cache_hmr2s_from_batches(
            model,
            len(track["paths"]),
            path_crop_batches(
                track["paths"], track["crop_box"], track["valid"], batch_size
            ),
            device,
            f"3dpw_val_{ordinal}",
        )
        track["hmr2s_token"] = cached.token
        track["hmr2s_pose"] = cached.pose
        track["hmr2s_betas"] = cached.betas


class Cached3DPWClips(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        labels: dict[str, Any],
        tracks: list[phase3.TrainTrack],
        observations: dict[str, np.ndarray],
        hmr: HMRCache,
        clip_frames: int,
        stride: int,
        maximum: int,
        seed: int,
    ) -> None:
        self.labels = labels
        self.observations = observations
        self.hmr = hmr
        clips: list[np.ndarray] = []
        for track in tracks:
            for start in range(0, len(track.indices) - clip_frames + 1, stride):
                rows = track.indices[start : start + clip_frames]
                valid = observations["valid"][rows]
                if bool(valid[0]) and float(valid.mean()) >= 0.75:
                    clips.append(rows.astype(np.int64))
        random.Random(seed).shuffle(clips)
        self.clips = clips[:maximum] if maximum > 0 else clips
        if not self.clips:
            raise RuntimeError("No 3DPW training clips survived the YOLO guard")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.clips[index]
        raw_pose = (
            wham_eval.to_numpy(phase3.take_rows(self.labels["pose"], rows))
            .astype(np.float32)
            .reshape(len(rows), 24, 3)
        )
        target_pose = wham_eval.matrix_to_rotation_6d(
            wham_eval.axis_angle_to_matrix(torch.from_numpy(raw_pose))
        )
        betas = wham_eval.to_numpy(phase3.take_rows(self.labels["betas"], rows)).astype(
            np.float32
        )[..., :10]
        return {
            "x": torch.from_numpy(self.observations["x"][rows].copy()),
            "mask": torch.from_numpy(self.observations["mask"][rows].copy()),
            "valid": torch.from_numpy(self.observations["valid"][rows].copy()),
            "token": torch.from_numpy(self.hmr.token[rows].astype(np.float32)),
            "hmr_pose": torch.from_numpy(self.hmr.pose[rows].astype(np.float32)),
            "hmr_betas": torch.from_numpy(self.hmr.betas[rows].astype(np.float32)),
            "target_pose": target_pose,
            "target_betas": torch.from_numpy(betas),
            "source": torch.tensor(0, dtype=torch.int64),
        }


class CachedBedlamClips(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[bedlam.BedlamSample],
        clips: list[list[bedlam.BedlamSample]],
        hmr: HMRCache,
    ) -> None:
        self.samples = samples
        self.hmr = hmr
        lookup = {id(sample): index for index, sample in enumerate(samples)}
        self.clips = [
            np.asarray([lookup[id(sample)] for sample in clip]) for clip in clips
        ]
        if not self.clips:
            raise RuntimeError("No contiguous BEDLAM clips were found")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        indices = self.clips[index]
        selected = [self.samples[int(row)] for row in indices]
        raw_pose = np.stack([sample.pose for sample in selected]).astype(np.float32)
        target_pose = wham_eval.matrix_to_rotation_6d(
            wham_eval.axis_angle_to_matrix(torch.from_numpy(raw_pose))
        )
        return {
            "x": torch.from_numpy(np.stack([sample.x for sample in selected])),
            "mask": torch.from_numpy(np.stack([sample.mask for sample in selected])),
            "valid": torch.ones(len(selected), dtype=torch.bool),
            "token": torch.from_numpy(self.hmr.token[indices].astype(np.float32)),
            "hmr_pose": torch.from_numpy(self.hmr.pose[indices].astype(np.float32)),
            "hmr_betas": torch.from_numpy(self.hmr.betas[indices].astype(np.float32)),
            "target_pose": target_pose,
            "target_betas": torch.from_numpy(
                np.stack([sample.shape for sample in selected]).astype(np.float32)
            ),
            "source": torch.tensor(1, dtype=torch.int64),
        }


def rotation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_matrix = wham_eval.rotation_6d_to_matrix(prediction.float())
    target_matrix = wham_eval.rotation_6d_to_matrix(target.float())
    relative = prediction_matrix @ target_matrix.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


def adapter_batch_losses(
    adapter: TokenAdapter,
    readout: PoseReadout,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = adapter(source)
    target_std = adapter.target_std.clamp_min(1e-4)
    standardized = F.smooth_l1_loss(
        (prediction - adapter.target_mean) / target_std,
        (target - adapter.target_mean) / target_std,
    )
    cosine = (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    ).mean()
    predicted_pose = readout(prediction.float())
    target_pose = readout(target.float())
    predicted_matrix = hmr2_rotation_6d_to_matrix(predicted_pose)
    target_matrix = hmr2_rotation_6d_to_matrix(target_pose)
    relative = predicted_matrix.transpose(-1, -2) @ target_matrix
    pose_cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    pose_skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    pose_sine = 0.5 * torch.linalg.vector_norm(pose_skew, dim=-1)
    pose = torch.atan2(pose_sine, pose_cosine).mean()
    total = standardized + 0.25 * cosine + 0.25 * pose
    return total, {
        "loss": total.detach(),
        "standardized_smooth_l1": standardized.detach(),
        "cosine_loss": cosine.detach(),
        "pose_rad": pose.detach(),
    }


def train_adapter_epoch(
    adapter: TokenAdapter,
    readout: PoseReadout,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    adapter.train()
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for source, target, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, parts = adapter_batch_losses(adapter, readout, source, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = len(source)
        samples += count
        for name, value in parts.items():
            totals[name] += float(value) * count
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.inference_mode()
def evaluate_adapter_pairs(
    adapter: TokenAdapter,
    readout: PoseReadout,
    source: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    for start in range(0, len(source), batch_size):
        source_batch = torch.from_numpy(
            source[start : start + batch_size].astype(np.float32)
        ).to(device)
        target_batch = torch.from_numpy(
            target[start : start + batch_size].astype(np.float32)
        ).to(device)
        _, parts = adapter_batch_losses(adapter, readout, source_batch, target_batch)
        count = len(source_batch)
        for name, value in parts.items():
            totals[name] += float(value) * count
    return {name: value / max(len(source), 1) for name, value in totals.items()}


def configure_wham_stage(network: nn.Module, stage: str) -> list[dict[str, Any]]:
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    groups: list[dict[str, Any]] = []
    fast_parameters: list[nn.Parameter] = []
    for module in (network.integrator, network.motion_decoder.neural_init):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            fast_parameters.append(parameter)
    groups.append(
        {"params": fast_parameters, "lr": 1.0e-5 if stage == "integration" else 3.0e-6}
    )
    if stage == "decoder":
        decoder_parameters: list[nn.Parameter] = []
        for parameter in network.motion_decoder.regressor.parameters():
            parameter.requires_grad_(True)
            decoder_parameters.append(parameter)
        groups.append({"params": decoder_parameters, "lr": 5.0e-7})
    elif stage != "integration":
        raise ValueError(stage)
    return groups


def set_wham_training_modes(network: nn.Module, stage: str) -> None:
    network.eval()
    network.integrator.train()
    network.motion_decoder.neural_init.train()
    # cuDNN RNN backward requires the reserve buffer even when its weights are frozen.
    network.motion_decoder.regressor.rnn.dropout = 0.0
    network.motion_decoder.regressor.rnn.train()
    if stage == "decoder":
        network.motion_decoder.regressor.train()


def native_wham_forward(
    network: nn.Module,
    initializer: frozen_eval.FrozenSMPLInitializer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["x"]
    mask = batch["mask"].bool()
    valid = batch["valid"].bool()
    token = batch["token"]
    hmr_pose = batch["hmr_pose"]
    hmr_betas = batch["hmr_betas"]
    with torch.no_grad():
        init_joints = initializer(hmr_pose[:, 0], hmr_betas[:, 0])
        init_kp = torch.cat((init_joints.reshape(len(x), 1, 51), x[:, :1]), dim=-1)
        processed = network.preprocess(x[:, 1:].clone(), mask[:, 1:])
        _, context = network.motion_encoder(processed, init_kp)
    integrated = deployment.integrate_deployment(
        network, context.detach(), token[:, 1:], valid[:, 1:]
    )
    predicted_pose, predicted_shape, _, _ = network.motion_decoder(
        integrated, hmr_pose[:, :1]
    )
    return (
        predicted_pose.reshape(len(x), x.shape[1] - 1, 24, 6),
        predicted_shape.reshape(len(x), x.shape[1] - 1, 10),
    )


def wham_tuning_loss(
    network: nn.Module,
    initializer: frozen_eval.FrozenSMPLInitializer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_pose, predicted_shape = native_wham_forward(network, initializer, batch)
    target_pose = batch["target_pose"][:, 1:]
    target_shape = batch["target_betas"][:, 1:]
    pose_per_joint = rotation_loss(predicted_pose, target_pose)
    body_pose = pose_per_joint[..., 1:].mean()
    root_pose = pose_per_joint[..., 0].mean()
    shape = F.smooth_l1_loss(predicted_shape, target_shape)
    if predicted_pose.shape[1] >= 3:
        pred_matrix = wham_eval.rotation_6d_to_matrix(predicted_pose.float())
        target_matrix = wham_eval.rotation_6d_to_matrix(target_pose.float())
        pred_velocity = pred_matrix[:, 1:] @ pred_matrix[:, :-1].transpose(-1, -2)
        target_velocity = target_matrix[:, 1:] @ target_matrix[:, :-1].transpose(-1, -2)
        temporal = rotation_loss(
            wham_eval.matrix_to_rotation_6d(pred_velocity),
            wham_eval.matrix_to_rotation_6d(target_velocity),
        ).mean()
    else:
        temporal = body_pose.new_zeros(())
    total = 4.0 * body_pose + 2.0 * root_pose + 0.4 * shape + 0.5 * temporal
    return total, {
        "loss": total.detach(),
        "body_pose_rad": body_pose.detach(),
        "root_pose_rad": root_pose.detach(),
        "shape_smooth_l1": shape.detach(),
        "temporal_rad": temporal.detach(),
    }


def train_wham_epoch(
    network: nn.Module,
    initializer: frozen_eval.FrozenSMPLInitializer,
    loaders: Sequence[DataLoader],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    iterators = [iter(loader) for loader in loaders]
    steps = max(len(loader) for loader in loaders)
    for step in range(steps):
        for loader_index, loader in enumerate(loaders):
            try:
                batch = next(iterators[loader_index])
            except StopIteration:
                iterators[loader_index] = iter(loader)
                batch = next(iterators[loader_index])
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, parts = wham_tuning_loss(network, initializer, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable = [
                parameter
                for parameter in network.parameters()
                if parameter.requires_grad
            ]
            nn.utils.clip_grad_norm_(trainable, 0.5)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["x"])
            samples += count
            for name, value in parts.items():
                totals[name] += float(value) * count
        if (step + 1) % 100 == 0 or step + 1 == steps:
            json_log(stage="wham_train", step=step + 1, steps=steps)
    return {name: value / max(samples, 1) for name, value in totals.items()}


def concatenate_summary(parts: list[np.ndarray]) -> dict[str, float | int]:
    if not parts:
        raise RuntimeError("No metric values were accumulated")
    return wham_eval.summarize(np.concatenate(parts))


@torch.inference_mode()
def evaluate_cached_validation(
    network: nn.Module,
    adapter: TokenAdapter | None,
    labels: dict[str, Any],
    tracks: list[dict[str, Any]],
    initializer: frozen_eval.FrozenSMPLInitializer,
    smpl_models: dict[str, nn.Module],
    h36m_regressor: torch.Tensor,
    device: torch.device,
    smpl_batch_size: int,
) -> dict[str, Any]:
    network.eval()
    if adapter is not None:
        adapter.eval()
    accumulated: dict[str, list[np.ndarray]] = defaultdict(list)
    detections = 0
    source_frames = 0
    completed = 0
    skipped: list[str] = []
    for track in tracks:
        index = int(track["index"])
        available = int(track["available"])
        source_frames += available + 1
        detections += int(np.asarray(track["valid"][: available + 1]).sum())
        if not bool(track["valid"][0]):
            skipped.append(str(track["video_id"]))
            continue
        token = torch.from_numpy(
            track["hmr2s_token"][1 : available + 1].astype(np.float32)
        ).to(device)
        if adapter is not None:
            token = adapter(token)
        first_pose = torch.from_numpy(track["hmr2s_pose"][:1].astype(np.float32)).to(
            device
        )
        first_betas = torch.from_numpy(track["hmr2s_betas"][:1].astype(np.float32)).to(
            device
        )
        init_joints = initializer(first_pose, first_betas).reshape(1, 1, 51)
        first_x = torch.from_numpy(track["x"][:1].astype(np.float32)).to(device)
        prediction = frozen_eval.run_phone_core(
            network=network,
            x=torch.from_numpy(track["x"][1 : available + 1].astype(np.float32))
            .unsqueeze(0)
            .to(device),
            mask=torch.from_numpy(track["mask"][1 : available + 1])
            .unsqueeze(0)
            .to(device),
            features=token.unsqueeze(0),
            feature_valid=torch.from_numpy(
                track["valid"][1 : available + 1].astype(np.float32)
            )
            .reshape(1, available, 1)
            .to(device),
            init_kp=torch.cat((init_joints, first_x.reshape(1, 1, 37)), dim=-1),
            init_pose=first_pose.reshape(1, 1, 24, 6),
            init_root=first_pose[:, 0].reshape(1, 1, 6),
            cam_angvel=torch.zeros(1, available, 6, device=device),
        )
        target_pose = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1 : available + 1].astype(
                np.float32
            )
        )
        target_betas = torch.from_numpy(
            wham_eval.to_numpy(labels["betas"][index])[1 : available + 1].astype(
                np.float32
            )
        )
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
        for name, values in metrics.items():
            accumulated[name].append(values)
        completed += 1
    if not completed:
        raise RuntimeError("No validation track had a valid initializer frame")
    return {
        "metrics": {
            name: concatenate_summary(values) for name, values in accumulated.items()
        },
        "tracks": completed,
        "source_frames": source_frames,
        "detections": detections,
        "detection_rate": detections / max(source_frames, 1),
        "skipped_initializer_tracks": skipped,
    }


def validation_score(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    weights = {
        "pa_mpjpe_mm": 0.40,
        "mpjpe_mm": 0.25,
        "pve_mm": 0.25,
        "accel_official_30fps": 0.10,
    }
    return sum(
        weight
        * mean_metric(candidate["metrics"], name)
        / max(mean_metric(baseline["metrics"], name), 1e-8)
        for name, weight in weights.items()
    )


def compact_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: mean_metric(result["metrics"], name)
        for name in (
            "pa_mpjpe_mm",
            "mpjpe_mm",
            "pve_mm",
            "accel_official_30fps",
        )
    }


def train_adapter(
    dataset: PairDataset,
    validation_source: np.ndarray,
    validation_target: np.ndarray,
    readout_values: dict[str, torch.Tensor],
    released_network: nn.Module,
    labels_val: dict[str, Any],
    validation_tracks: list[dict[str, Any]],
    initializer: frozen_eval.FrozenSMPLInitializer,
    smpl_models: dict[str, nn.Module],
    h36m_regressor: torch.Tensor,
    output_path: Path,
    epochs: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    smpl_batch_size: int,
) -> tuple[TokenAdapter, list[dict[str, Any]], dict[str, Any]]:
    adapter = TokenAdapter().to(device)
    target = dataset.target.astype(np.float32)
    adapter.target_mean.copy_(torch.from_numpy(target.mean(axis=0)).to(device))
    adapter.target_std.copy_(
        torch.from_numpy(target.std(axis=0).clip(min=1e-4)).to(device)
    )
    readout = PoseReadout(readout_values).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2.0e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1.0e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    baseline = evaluate_cached_validation(
        released_network,
        None,
        labels_val,
        validation_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        smpl_batch_size,
    )
    best_score = 1.0
    best_epoch = 0
    best_state = copy.deepcopy(adapter.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = train_adapter_epoch(
            adapter, readout, loader, optimizer, scaler, device
        )
        pair_validation = evaluate_adapter_pairs(
            adapter,
            readout,
            validation_source,
            validation_target,
            batch_size,
            device,
        )
        end_to_end = evaluate_cached_validation(
            released_network,
            adapter,
            labels_val,
            validation_tracks,
            initializer,
            smpl_models,
            h36m_regressor,
            device,
            smpl_batch_size,
        )
        score = validation_score(end_to_end, baseline)
        accepted = math.isfinite(score) and score < best_score
        if accepted:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
        row = {
            "candidate": "adapter",
            "stage": "token_adapter",
            "epoch": epoch,
            "training": training,
            "pair_validation": pair_validation,
            "end_to_end_validation": compact_metrics(end_to_end),
            "validation_score_vs_naive": score,
            "selected_so_far": accepted,
            "minutes": (time.perf_counter() - started) / 60.0,
        }
        history.append(row)
        json_log(**row)
        scheduler.step()
    adapter.load_state_dict(best_state, strict=True)
    selected_validation = evaluate_cached_validation(
        released_network,
        adapter if best_epoch > 0 else None,
        labels_val,
        validation_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        smpl_batch_size,
    )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "candidate": "hmr2s_to_hmr2a_token_adapter",
        "architecture": "residual_layernorm_mlp_1024_1024_1024",
        "adapter_state_dict": best_state,
        "selected_epoch": best_epoch,
        "uses_learned_adapter": best_epoch > 0,
        "validation": selected_validation,
        "baseline_validation": baseline,
    }
    torch.save(checkpoint, output_path)
    summary = {
        "selected_epoch": best_epoch,
        "uses_learned_adapter": best_epoch > 0,
        "baseline": compact_metrics(baseline),
        "selected": compact_metrics(selected_validation),
        "score_vs_naive": validation_score(selected_validation, baseline),
        "checkpoint": str(output_path),
    }
    return adapter, history, summary


def train_native_wham(
    source_network: nn.Module,
    clip_datasets: Sequence[Dataset[dict[str, torch.Tensor]]],
    labels_val: dict[str, Any],
    validation_tracks: list[dict[str, Any]],
    initializer: frozen_eval.FrozenSMPLInitializer,
    smpl_models: dict[str, nn.Module],
    h36m_regressor: torch.Tensor,
    output_path: Path,
    integration_epochs: int,
    decoder_epochs: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    smpl_batch_size: int,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    network = copy.deepcopy(source_network).to(device)
    baseline = evaluate_cached_validation(
        source_network,
        None,
        labels_val,
        validation_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        smpl_batch_size,
    )
    best_score = 1.0
    best_epoch = 0
    best_stage = "released_baseline"
    best_state = copy.deepcopy(network.state_dict())
    history: list[dict[str, Any]] = []
    global_epoch = 0
    for stage, epochs in (
        ("integration", integration_epochs),
        ("decoder", decoder_epochs),
    ):
        groups = configure_wham_stage(network, stage)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-7
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                persistent_workers=workers > 0,
                drop_last=len(dataset) >= batch_size,
            )
            for dataset in clip_datasets
        ]
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            started = time.perf_counter()
            set_wham_training_modes(network, stage)
            training = train_wham_epoch(
                network, initializer, loaders, optimizer, scaler, device
            )
            validation = evaluate_cached_validation(
                network,
                None,
                labels_val,
                validation_tracks,
                initializer,
                smpl_models,
                h36m_regressor,
                device,
                smpl_batch_size,
            )
            score = validation_score(validation, baseline)
            accepted = math.isfinite(score) and score < best_score
            if accepted:
                best_score = score
                best_epoch = global_epoch
                best_stage = stage
                best_state = copy.deepcopy(network.state_dict())
            row = {
                "candidate": "native_wham",
                "stage": stage,
                "stage_epoch": stage_epoch,
                "epoch": global_epoch,
                "training": training,
                "end_to_end_validation": compact_metrics(validation),
                "validation_score_vs_naive": score,
                "selected_so_far": accepted,
                "minutes": (time.perf_counter() - started) / 60.0,
            }
            history.append(row)
            json_log(**row)
            scheduler.step()
    network.load_state_dict(best_state, strict=True)
    selected_validation = evaluate_cached_validation(
        network,
        None,
        labels_val,
        validation_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        smpl_batch_size,
    )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "candidate": "native_hmr2s_tuned_wham",
        "wham_state_dict": best_state,
        "selected_epoch": best_epoch,
        "selected_stage": best_stage,
        "uses_tuned_wham": best_epoch > 0,
        "validation": selected_validation,
        "baseline_validation": baseline,
    }
    torch.save(checkpoint, output_path)
    summary = {
        "selected_epoch": best_epoch,
        "selected_stage": best_stage,
        "uses_tuned_wham": best_epoch > 0,
        "baseline": compact_metrics(baseline),
        "selected": compact_metrics(selected_validation),
        "score_vs_naive": validation_score(selected_validation, baseline),
        "checkpoint": str(output_path),
    }
    return network, history, summary


def load_bedlam_samples(
    label_root: Path,
    hf_repo: str,
    token: str,
    pose_model: YOLO,
    scratch_dir: Path,
    hf_cache_dir: Path,
    device: torch.device,
    maximum_scenes: int,
    maximum_download_gib: float,
    videos_per_scene: int,
    frames_per_video: int,
    yolo_batch_size: int,
) -> tuple[list[bedlam.BedlamSample], dict[str, Any]]:
    label_paths = [
        path for path in sorted(label_root.glob("*.npz")) if path.name != "agora.npz"
    ]
    if len(label_paths) < 20:
        raise RuntimeError(
            f"BEDLAM label directory {label_root} contains only {len(label_paths)} npz files"
        )
    for path in label_paths:
        bedlam.validate_label_file(path)
    scenes, unmatched = bedlam.resolve_remote_scenes(label_paths, hf_repo, token)
    selected = bedlam.select_scenes(scenes, maximum_scenes, maximum_download_gib)
    all_samples: list[bedlam.BedlamSample] = []
    scene_rows: list[dict[str, Any]] = []
    scratch_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    for scene_ordinal, scene in enumerate(selected, start=1):
        json_log(
            stage="bedlam_download",
            scene=f"{scene_ordinal}/{len(selected)}",
            remote=scene.remote_path,
            gib=scene.remote_size / (1024**3),
        )
        tar_path = Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=scene.remote_path,
                repo_type="dataset",
                revision="main",
                token=token,
                cache_dir=hf_cache_dir,
            )
        )
        accepted_before = len(all_samples)
        attempted = 0
        with (
            np.load(scene.label_path, allow_pickle=False) as archive_labels,
            tarfile.open(tar_path, mode="r:*") as archive,
        ):
            labels = {
                key: archive_labels[key]
                for key in ("imgname", "center", "scale", "pose_cam", "shape", "gender")
            }
            sequence_rows = bedlam.map_sequence_rows(labels["imgname"])
            members = bedlam.tar_video_members(archive)
            available = sorted(
                sequence
                for sequence in sequence_rows
                if bedlam.locate_tar_member(members, sequence) is not None
            )
            chosen = bedlam.evenly_spaced(available, videos_per_scene)
            for sequence in chosen:
                member = bedlam.locate_tar_member(members, sequence)
                if member is None:
                    continue
                video_path = scratch_dir / "active_bedlam_video.mp4"
                bedlam.extract_member(archive, member, video_path)
                try:
                    samples, statistics = bedlam.samples_from_video(
                        labels,
                        sequence_rows[sequence],
                        video_path,
                        pose_model,
                        device,
                        frames_per_video,
                        yolo_batch_size,
                        rotate_clockwise="closeup" in scene.label_scene.lower(),
                        confidence_threshold=0.5,
                        minimum_iou=0.45,
                    )
                finally:
                    video_path.unlink(missing_ok=True)
                attempted += int(statistics["attempted_person_frames"])
                all_samples.extend(samples)
        accepted = len(all_samples) - accepted_before
        scene_rows.append(
            {
                "scene": scene.label_scene,
                "remote": scene.remote_path,
                "download_gib": scene.remote_size / (1024**3),
                "attempted_person_frames": attempted,
                "accepted_person_frames": accepted,
                "detection_match_rate": accepted / max(attempted, 1),
            }
        )
        json_log(stage="bedlam_scene_complete", **scene_rows[-1])
    if not all_samples:
        raise RuntimeError("No BEDLAM sample survived YOLO-to-label matching")
    return all_samples, {
        "hf_repo": hf_repo,
        "labels_available": len(label_paths),
        "matched_archives": len(scenes),
        "unmatched_labels": unmatched,
        "selected_scenes": scene_rows,
        "accepted_samples": len(all_samples),
        "yolo_confidence_threshold": 0.5,
        "label_match_minimum_iou": 0.45,
    }


def single_person_test_indices(labels: dict[str, Any], image_root: Path) -> list[int]:
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
    adapter: TokenAdapter | None,
    observations: dict[str, Any],
    initializer: frozen_eval.FrozenSMPLInitializer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    frames = int(observations["frames"]) - 1
    first_pose = observations["pose"][:1].to(device)
    first_betas = observations["betas"][:1].to(device)
    init_joints = initializer(first_pose, first_betas).reshape(1, 1, 51)
    token = observations["token"][1:].to(device)
    if adapter is not None:
        token = adapter(token)
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


@torch.inference_mode()
def final_test_evaluation(
    labels: dict[str, Any],
    image_root: Path,
    wham_repo: Path,
    released_network: nn.Module,
    tuned_network: nn.Module,
    adapter: TokenAdapter,
    adapter_enabled: bool,
    pose_model: YOLO,
    hmr2s: FrozenHMR2S,
    initializer: frozen_eval.FrozenSMPLInitializer,
    smpl_models: dict[str, nn.Module],
    h36m_regressor: torch.Tensor,
    device: torch.device,
    pose_batch_size: int,
    hmr_batch_size: int,
    smpl_batch_size: int,
    output_json: Path,
    output_csv: Path,
    yolo_variant: str = "yolo26n-pose",
) -> dict[str, Any]:
    selected = single_person_test_indices(labels, image_root)
    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        variant: defaultdict(list) for variant in FINAL_VARIANTS
    }
    timings: dict[str, defaultdict[str, float]] = {
        variant: defaultdict(float) for variant in FINAL_VARIANTS
    }
    rows: list[dict[str, Any]] = []
    detections = 0
    source_frames = 0
    skipped: list[dict[str, str]] = []
    for ordinal, index in enumerate(selected, start=1):
        video_id = str(labels["vid"][index])
        frames = frozen_eval.available_frames(labels, index)
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

        normal = reference_eval.official_inputs(labels, index, frames, "", device)
        flipped = reference_eval.official_inputs(
            labels, index, frames, "flipped_", device
        )
        synchronize(device)
        started = time.perf_counter()
        released_output = reference_eval.average_flipped_prediction(
            wham_repo,
            reference_eval.run_core(released_network, normal),
            reference_eval.run_core(released_network, flipped),
        )
        synchronize(device)
        timings["released_wham"]["wham_core_and_flip"] += time.perf_counter() - started
        phone_predictions = {
            "naive_hmr2s_released_wham": run_phone_variant(
                released_network, None, observations, initializer, device
            ),
            "adapter_hmr2s_released_wham": run_phone_variant(
                released_network,
                adapter if adapter_enabled else None,
                observations,
                initializer,
                device,
            ),
            "native_hmr2s_tuned_wham": run_phone_variant(
                tuned_network, None, observations, initializer, device
            ),
        }
        predictions = {"released_wham": released_output, **phone_predictions}
        target_pose = torch.from_numpy(
            wham_eval.to_numpy(labels["pose"][index])[1 : frames + 1].astype(np.float32)
        )
        target_betas = torch.from_numpy(
            wham_eval.to_numpy(labels["betas"][index])[1 : frames + 1].astype(
                np.float32
            )
        )
        for variant, prediction in predictions.items():
            metrics, decode_seconds = reference_eval.smpl_metrics(
                prediction,
                target_pose,
                target_betas,
                str(labels["gender"][index]).lower(),
                smpl_models,
                h36m_regressor,
                device,
                smpl_batch_size,
            )
            timings[variant]["smpl_metric_decode"] += decode_seconds
            row: dict[str, Any] = {
                "variant": variant,
                "sequence": video_id,
                "frames": frames,
                "detections": int(observations["detections"]),
                "detection_rate": int(observations["detections"]) / len(paths),
            }
            for name, values in metrics.items():
                accumulated[variant][name].append(values)
                row[name] = float(values.mean()) if len(values) else None
            rows.append(row)
        json_log(
            stage="final_test",
            track=f"{ordinal}/{len(selected)}",
            sequence=video_id,
            frames=frames,
            pa_mpjpe_mm={row["variant"]: row["pa_mpjpe_mm"] for row in rows[-4:]},
        )
    if not rows:
        raise RuntimeError("No 3DPW test track completed the four-way comparison")
    variants = {
        variant: {
            "metrics": {
                name: concatenate_summary(parts)
                for name, parts in accumulated[variant].items()
            },
            "work_seconds": dict(timings[variant]),
        }
        for variant in FINAL_VARIANTS
    }
    released = variants["released_wham"]["metrics"]
    for variant in FINAL_VARIANTS[1:]:
        variants[variant]["minus_released"] = {
            metric: mean_metric(variants[variant]["metrics"], metric)
            - mean_metric(released, metric)
            for metric in released
        }
        variants[variant]["relative_change_vs_released"] = {
            metric: mean_metric(variants[variant]["metrics"], metric)
            / max(mean_metric(released, metric), 1e-8)
            - 1.0
            for metric in released
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "locked_hmr2s_adapter_vs_native_wham_tuning",
        "protocol": {
            "dataset": "3DPW test",
            "population": "all matching single-person tracks with a valid YOLO26 initializer detection",
            "test_used_for_training_or_selection": False,
            "same_population": True,
            "camera_signal": "zero angular velocity because 3DPW has no phone gyroscope stream",
            "released_wham": "official stored ViTPose/HMR2a inputs, flip average, released WHAM",
            "phone_frontend": f"{yolo_variant} + released HMR2.0-S pose/shape initialization and image token",
        },
        "population": {
            "eligible_tracks": len(selected),
            "completed_tracks": len({row["sequence"] for row in rows}),
            "recurrent_frames_per_variant": variants["released_wham"]["metrics"][
                "mpjpe_mm"
            ]["samples"],
            "source_frames": source_frames,
            "detections": detections,
            "detection_rate": detections / max(source_frames, 1),
            "skipped": skipped,
        },
        "variants": variants,
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    return report


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    output[f"{key}.{child_key}"] = child_value
            else:
                output[key] = value
        flattened.append(output)
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def self_test() -> None:
    torch.manual_seed(1)
    adapter = TokenAdapter(hidden_dim=32)
    token = torch.randn(3, 1024)
    assert torch.equal(adapter(token), token)
    candidate = {
        "metrics": {
            "pa_mpjpe_mm": {"mean": 50.0},
            "mpjpe_mm": {"mean": 60.0},
            "pve_mm": {"mean": 70.0},
            "accel_official_30fps": {"mean": 8.0},
        }
    }
    baseline = {
        "metrics": {
            "pa_mpjpe_mm": {"mean": 100.0},
            "mpjpe_mm": {"mean": 120.0},
            "pve_mm": {"mean": 140.0},
            "accel_official_30fps": {"mean": 16.0},
        }
    }
    assert abs(validation_score(candidate, baseline) - 0.5) < 1e-7
    assert set(FINAL_VARIANTS) == {
        "released_wham",
        "naive_hmr2s_released_wham",
        "adapter_hmr2s_released_wham",
        "native_hmr2s_tuned_wham",
    }

    class FakeMotionEncoder(nn.Module):
        def forward(
            self, value: torch.Tensor, initial: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del initial
            batch, frames = value.shape[:2]
            return (
                torch.zeros(batch, frames, 17, 3),
                torch.zeros(batch, frames, 563),
            )

    class FakeRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(563, 16, batch_first=True)

    class FakeMotionDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.neural_init = nn.Linear(144, 16)
            self.regressor = FakeRegressor()
            self.pose = nn.Linear(563, 144)
            self.shape = nn.Linear(563, 10)

        def forward(
            self, value: torch.Tensor, initial: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del initial
            batch, frames = value.shape[:2]
            return (
                self.pose(value).reshape(batch, frames, 24, 6),
                self.shape(value),
                torch.zeros(batch, frames, 3),
                torch.zeros(batch, frames, 4),
            )

    class FakeIntegrator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(1587, 563)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(563, 563)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(563, 563)

    class FakeNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.motion_encoder = FakeMotionEncoder()
            self.integrator = FakeIntegrator()
            self.motion_decoder = FakeMotionDecoder()

        @staticmethod
        def preprocess(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            del mask
            return value

    class FakeInitializer(nn.Module):
        @staticmethod
        def forward(pose: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
            del betas
            return torch.zeros(len(pose), 17, 3)

    identity = wham_eval.matrix_to_rotation_6d(torch.eye(3)).repeat(2, 13, 24, 1)
    fake_batch = {
        "x": torch.zeros(2, 13, 37),
        "mask": torch.zeros(2, 13, 17, dtype=torch.bool),
        "valid": torch.ones(2, 13, dtype=torch.bool),
        "token": torch.randn(2, 13, 1024),
        "hmr_pose": identity.clone(),
        "hmr_betas": torch.zeros(2, 13, 10),
        "target_pose": identity.clone(),
        "target_betas": torch.zeros(2, 13, 10),
    }
    fake_network = FakeNetwork()
    fake_loss, _ = wham_tuning_loss(fake_network, FakeInitializer(), fake_batch)
    fake_loss.backward()
    assert fake_network.integrator.layer1.weight.grad is not None
    print(
        "Self-test passed: adapter identity, locked selection, final variants, "
        "and native-token WHAM gradient path"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--bedlam-label-root", type=Path)
    parser.add_argument("--three-dpw-root", type=Path)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--train-parsed", type=Path)
    parser.add_argument("--val-parsed", type=Path)
    parser.add_argument("--test-parsed", type=Path)
    parser.add_argument("--wham-repo", type=Path)
    parser.add_argument("--wham-checkpoint", type=Path)
    parser.add_argument("--hmr2a-checkpoint", type=Path)
    parser.add_argument("--hmr2s-repo", type=Path)
    parser.add_argument("--hmr2s-checkpoint", type=Path)
    parser.add_argument("--yolo26-weights", type=Path)
    parser.add_argument("--smpl-model-directory", type=Path)
    parser.add_argument("--h36m-joint-regressor", type=Path)
    parser.add_argument("--wham-joint-regressor", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--hf-repo", default=bedlam.HF_REPO_DEFAULT)
    parser.add_argument("--yolo-variant", default="yolo26n-pose")
    parser.add_argument("--expected-yolo-sha256", default=deployment.YOLO26_SHA256)
    parser.add_argument("--coco-train-people", type=int, default=20000)
    parser.add_argument("--coco-val-people", type=int, default=2000)
    parser.add_argument("--maximum-bedlam-scenes", type=int, default=10)
    parser.add_argument("--maximum-bedlam-download-gib", type=float, default=4.0)
    parser.add_argument("--bedlam-videos-per-scene", type=int, default=24)
    parser.add_argument("--bedlam-frames-per-video", type=int, default=200)
    parser.add_argument("--maximum-bedlam-clips", type=int, default=1500)
    parser.add_argument("--maximum-3dpw-clips", type=int, default=1500)
    parser.add_argument("--clip-frames", type=int, default=13)
    parser.add_argument("--clip-stride", type=int, default=6)
    parser.add_argument("--validation-tracks", type=int, default=12)
    parser.add_argument("--validation-frames", type=int, default=300)
    parser.add_argument("--adapter-epochs", type=int, default=6)
    parser.add_argument("--integration-epochs", type=int, default=3)
    parser.add_argument("--decoder-epochs", type=int, default=3)
    parser.add_argument("--pose-batch-size", type=int, default=24)
    parser.add_argument("--teacher-batch-size", type=int, default=12)
    parser.add_argument("--hmr-batch-size", type=int, default=24)
    parser.add_argument("--adapter-batch-size", type=int, default=256)
    parser.add_argument("--wham-batch-size", type=int, default=2)
    parser.add_argument("--smpl-batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def require_main_paths(args: argparse.Namespace) -> None:
    path_names = (
        "coco_root",
        "bedlam_label_root",
        "three_dpw_root",
        "sequence_root",
        "train_parsed",
        "val_parsed",
        "test_parsed",
        "wham_repo",
        "wham_checkpoint",
        "hmr2a_checkpoint",
        "hmr2s_repo",
        "hmr2s_checkpoint",
        "yolo26_weights",
        "smpl_model_directory",
        "h36m_joint_regressor",
        "wham_joint_regressor",
        "output_dir",
        "scratch_dir",
    )
    absent = [name for name in path_names if getattr(args, name) is None]
    if absent:
        raise ValueError("Missing arguments: " + ", ".join(absent))
    required_files = (
        args.train_parsed,
        args.val_parsed,
        args.test_parsed,
        args.wham_checkpoint,
        args.hmr2a_checkpoint,
        args.hmr2s_checkpoint,
        args.yolo26_weights,
        args.h36m_joint_regressor,
        args.wham_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
        args.hmr2s_repo / "4D-Humans/hmr2/models/backbones/vit.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    for gender in ("NEUTRAL", "MALE", "FEMALE"):
        path = args.smpl_model_directory / f"SMPL_{gender}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    require_main_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing; add the gated BEDLAM token as a Kaggle secret"
        )
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.scratch_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    hmr2s_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.hmr2s_repo, text=True
    ).strip()
    if wham_commit != wham_eval.WHAM_COMMIT:
        raise RuntimeError(f"Wrong WHAM commit: {wham_commit}")
    from hmr2s_frozen import HMR2S_CHECKPOINT_SHA256, HMR2S_COMMIT

    if hmr2s_commit != HMR2S_COMMIT:
        raise RuntimeError(f"Wrong HMR2-S commit: {hmr2s_commit}")
    if sha256(args.hmr2s_checkpoint) != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError("Wrong HMR2-S checkpoint")
    if sha256(args.yolo26_weights) != args.expected_yolo_sha256:
        raise RuntimeError(f"Wrong {args.yolo_variant} checkpoint")

    if args.inspect_data:
        image_root = wham_eval.locate_image_root(args.three_dpw_root)
        train_labels = joblib.load(args.train_parsed)
        val_labels = joblib.load(args.val_parsed)
        sequence_root = phase3.locate_sequence_train_root(args.sequence_root)
        mapped_tracks = phase3.map_training_tracks(train_labels, sequence_root)
        val_indices = deployment.validation_indices(
            val_labels, image_root, args.validation_tracks
        )
        coco_train_images, coco_train_annotations = discover_split(
            args.coco_root, "train"
        )
        coco_val_images, coco_val_annotations = discover_split(args.coco_root, "val")
        label_paths = [
            path
            for path in sorted(args.bedlam_label_root.glob("*.npz"))
            if path.name != "agora.npz"
        ]
        if len(label_paths) < 20:
            raise RuntimeError(f"Only {len(label_paths)} BEDLAM label archives found")
        scenes, unmatched = bedlam.resolve_remote_scenes(
            label_paths, args.hf_repo, token
        )
        selected = bedlam.select_scenes(
            scenes,
            args.maximum_bedlam_scenes,
            args.maximum_bedlam_download_gib,
        )
        json_log(
            preflight="passed",
            three_dpw_train_rows=len(wham_eval.to_numpy(train_labels["vid"])),
            mapped_train_tracks=len(mapped_tracks),
            validation_tracks=len(val_indices),
            coco_train_images=coco_train_images,
            coco_train_annotations=coco_train_annotations,
            coco_val_images=coco_val_images,
            coco_val_annotations=coco_val_annotations,
            bedlam_labels=len(label_paths),
            bedlam_matched_scenes=len(scenes),
            bedlam_unmatched=len(unmatched),
            bedlam_selected=[scene.remote_path for scene in selected],
            selected_download_gib=sum(scene.remote_size for scene in selected)
            / (1024**3),
        )
        return

    image_root = wham_eval.locate_image_root(args.three_dpw_root)
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    train_sequence_root = phase3.locate_sequence_train_root(args.sequence_root)
    train_tracks = phase3.map_training_tracks(train_labels, train_sequence_root)
    train_paths = deployment.training_row_paths(train_labels, train_tracks, image_root)
    val_indices = deployment.validation_indices(
        val_labels, image_root, args.validation_tracks
    )
    train_images, train_annotations = discover_split(args.coco_root, "train")
    val_images, val_annotations = discover_split(args.coco_root, "val")
    coco_train_records = build_records(
        train_annotations, args.coco_train_people, args.seed
    )
    coco_val_records = build_records(
        val_annotations, args.coco_val_people, args.seed + 1
    )
    validate_record_images(train_images, coco_train_records)
    validate_record_images(val_images, coco_val_records)
    coco_train_dataset = PersonCropDataset(train_images, coco_train_records)
    coco_val_dataset = PersonCropDataset(val_images, coco_val_records)

    print(
        "Caching deployment YOLO26 observations for 3DPW train/validation...",
        flush=True,
    )
    pose_model = YOLO(str(args.yolo26_weights))
    train_observations = deployment.predict_observations(
        pose_model,
        train_paths,
        phase3.compact_bbox(train_labels["bbox"]),
        args.pose_batch_size,
        device,
        "3dpw_train",
        confidence_threshold=0.5,
        minimum_iou=0.45,
    )
    val_tracks = deployment.cache_validation_observations(
        cache_dir / "3dpw_val_yolo26.pth",
        val_labels,
        val_indices,
        image_root,
        pose_model,
        args.pose_batch_size,
        args.validation_frames,
        device,
        {
            "schema": SCHEMA_VERSION,
            "parsed_sha256": sha256(args.val_parsed),
            "yolo_sha256": args.expected_yolo_sha256,
            "indices": val_indices,
            "max_frames": args.validation_frames,
        },
    )
    bedlam_samples, bedlam_manifest = load_bedlam_samples(
        args.bedlam_label_root,
        args.hf_repo,
        token,
        pose_model,
        args.scratch_dir / "bedlam",
        args.hf_cache_dir or (args.scratch_dir / "hf_cache"),
        device,
        args.maximum_bedlam_scenes,
        args.maximum_bedlam_download_gib,
        args.bedlam_videos_per_scene,
        args.bedlam_frames_per_video,
        args.pose_batch_size,
    )
    del pose_model
    release_cuda()

    print("Caching HMR2a teacher tokens on COCO, 3DPW, and BEDLAM crops...", flush=True)
    teacher, readout_values = load_teacher(
        args.wham_repo, args.hmr2a_checkpoint, device
    )
    coco_teacher_train = cache_coco_teacher(
        teacher,
        coco_train_dataset,
        args.teacher_batch_size,
        args.workers,
        device,
        "coco_train",
    )
    coco_teacher_val = cache_coco_teacher(
        teacher,
        coco_val_dataset,
        args.teacher_batch_size,
        args.workers,
        device,
        "coco_val",
    )
    three_dpw_teacher = cache_teacher_from_batches(
        teacher,
        len(train_paths),
        path_crop_batches(
            train_paths,
            train_observations["crop_box"],
            train_observations["valid"],
            args.teacher_batch_size,
        ),
        device,
        "3dpw_train",
    )
    bedlam_teacher = cache_teacher_from_batches(
        teacher,
        len(bedlam_samples),
        bedlam_crop_batches(bedlam_samples, args.teacher_batch_size),
        device,
        "bedlam_train",
    )
    del teacher
    release_cuda()

    print("Caching frozen released HMR2-S outputs on the same crops...", flush=True)
    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    coco_hmr_train = cache_coco_hmr2s(
        hmr2s,
        coco_train_dataset,
        args.hmr_batch_size,
        args.workers,
        device,
        "coco_train",
    )
    coco_hmr_val = cache_coco_hmr2s(
        hmr2s,
        coco_val_dataset,
        args.hmr_batch_size,
        args.workers,
        device,
        "coco_val",
    )
    three_dpw_hmr = cache_hmr2s_from_batches(
        hmr2s,
        len(train_paths),
        path_crop_batches(
            train_paths,
            train_observations["crop_box"],
            train_observations["valid"],
            args.hmr_batch_size,
        ),
        device,
        "3dpw_train",
    )
    bedlam_hmr = cache_hmr2s_from_batches(
        hmr2s,
        len(bedlam_samples),
        bedlam_crop_batches(bedlam_samples, args.hmr_batch_size),
        device,
        "bedlam_train",
    )
    cache_validation_hmr2s(hmr2s, val_tracks, args.hmr_batch_size, device)
    del hmr2s
    release_cuda()

    valid_train = np.flatnonzero(train_observations["valid"])
    pair_dataset = PairDataset(
        (
            coco_hmr_train.token,
            three_dpw_hmr.token[valid_train],
            bedlam_hmr.token,
        ),
        (
            coco_teacher_train,
            three_dpw_teacher[valid_train],
            bedlam_teacher,
        ),
        PAIR_DOMAINS,
    )
    bedlam_clips = bedlam.build_temporal_clips(
        bedlam_samples,
        args.clip_frames,
        args.clip_stride,
        args.maximum_bedlam_clips,
    )
    clip_datasets: list[Dataset[dict[str, torch.Tensor]]] = [
        Cached3DPWClips(
            train_labels,
            train_tracks,
            train_observations,
            three_dpw_hmr,
            args.clip_frames,
            args.clip_stride,
            args.maximum_3dpw_clips,
            args.seed,
        ),
        CachedBedlamClips(bedlam_samples, bedlam_clips, bedlam_hmr),
    ]

    released_network = wham_eval.load_wham_core(
        args.wham_repo, args.wham_checkpoint, device
    )
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

    adapter_path = args.output_dir / "hmr2s_to_hmr2a_adapter_best.pth"
    adapter, adapter_history, adapter_summary = train_adapter(
        pair_dataset,
        coco_hmr_val.token,
        coco_teacher_val,
        readout_values,
        released_network,
        val_labels,
        val_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        adapter_path,
        args.adapter_epochs,
        args.adapter_batch_size,
        args.workers,
        device,
        args.smpl_batch_size,
    )
    tuned_path = args.output_dir / "hmr2s_native_wham_finetuned_best.pth"
    tuned_network, tuned_history, tuned_summary = train_native_wham(
        released_network,
        clip_datasets,
        val_labels,
        val_tracks,
        initializer,
        smpl_models,
        h36m_regressor,
        tuned_path,
        args.integration_epochs,
        args.decoder_epochs,
        args.wham_batch_size,
        args.workers,
        device,
        args.smpl_batch_size,
    )

    if adapter_summary["score_vs_naive"] <= tuned_summary["score_vs_naive"]:
        validation_choice = "adapter_hmr2s_released_wham"
        validation_choice_score = adapter_summary["score_vs_naive"]
    else:
        validation_choice = "native_hmr2s_tuned_wham"
        validation_choice_score = tuned_summary["score_vs_naive"]
    if validation_choice_score >= 1.0:
        validation_choice = "naive_hmr2s_released_wham"

    history = adapter_history + tuned_history
    write_history(args.output_dir / "hmr2s_wham_training_history.csv", history)
    training_report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "hmr2s_wham_token_adapter_and_native_finetune",
        "protocol": {
            "candidate_selection_split": "3DPW validation",
            "final_reporting_split": (
                "not opened in this validation-only grid run"
                if args.skip_final_test
                else "3DPW test opened once after selection"
            ),
            "test_leakage": False,
            "branch_independence": "both branches start from released WHAM; branch B never consumes branch A",
            "validation_selected_candidate": validation_choice,
            "selection_rule": "lowest predeclared normalized validation score: 40% PA-MPJPE, 25% MPJPE, 25% PVE, 10% acceleration",
            "yolo_variant": args.yolo_variant,
        },
        "data": {
            "coco": {
                "role": "same-crop HMR2-S/HMR2a token alignment for candidate A only",
                "train_people": len(coco_train_records),
                "validation_people": len(coco_val_records),
            },
            "three_dpw_train": {
                "role": "token alignment for A and temporal SMPL supervision for B",
                "rows": len(train_paths),
                "valid_yolo_rows": len(valid_train),
                "clips": len(clip_datasets[0]),
            },
            "bedlam_train": {
                "role": "token alignment for A and temporal SMPL supervision for B",
                **bedlam_manifest,
                "clips": len(clip_datasets[1]),
            },
            "three_dpw_validation": {
                "role": "checkpoint and candidate selection only",
                "tracks_requested": len(val_tracks),
            },
            "three_dpw_test": {
                "role": (
                    "withheld from this validation-only grid run"
                    if args.skip_final_test
                    else "one locked final evaluation only"
                ),
                "metrics": "PA-MPJPE, MPJPE, PVE, acceleration error",
            },
        },
        "candidate_2_adapter": adapter_summary,
        "candidate_3_native_wham_tuning": tuned_summary,
        "final_test_summary": None,
        "provenance": {
            "wham_commit": wham_commit,
            "hmr2s_commit": hmr2s_commit,
            "wham_checkpoint_sha256": sha256(args.wham_checkpoint),
            "hmr2a_checkpoint_sha256": sha256(args.hmr2a_checkpoint),
            "hmr2s_checkpoint_sha256": sha256(args.hmr2s_checkpoint),
            "yolo_variant": args.yolo_variant,
            "yolo26_checkpoint_sha256": sha256(args.yolo26_weights),
            "train_parsed_sha256": sha256(args.train_parsed),
            "val_parsed_sha256": sha256(args.val_parsed),
            "test_parsed_sha256": (
                None if args.skip_final_test else sha256(args.test_parsed)
            ),
        },
    }
    training_report_path = args.output_dir / "hmr2s_wham_training_report.json"
    if args.skip_final_test:
        training_report_path.write_text(
            json.dumps(training_report, indent=2) + "\n", encoding="utf-8"
        )
        json_log(
            completed=True,
            validation_only=True,
            yolo_variant=args.yolo_variant,
            validation_selected_candidate=validation_choice,
            validation_selected_score=validation_choice_score,
            output_dir=args.output_dir,
        )
        return

    # Deliberate protocol boundary: the test file is first loaded here, after both
    # candidates and the deployment choice have been locked on validation.
    print(
        "Candidates locked. Opening 3DPW test once for final reporting...", flush=True
    )
    test_labels = joblib.load(args.test_parsed)
    pose_model = YOLO(str(args.yolo26_weights))
    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    test_report = final_test_evaluation(
        test_labels,
        image_root,
        args.wham_repo,
        released_network,
        tuned_network,
        adapter,
        bool(adapter_summary["uses_learned_adapter"]),
        pose_model,
        hmr2s,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        args.pose_batch_size,
        args.hmr_batch_size,
        args.smpl_batch_size,
        args.output_dir / "hmr2s_wham_final_3dpw.json",
        args.output_dir / "hmr2s_wham_final_3dpw.csv",
        yolo_variant=args.yolo_variant,
    )
    training_report["final_test_summary"] = {
        variant: compact_metrics({"metrics": row["metrics"]})
        for variant, row in test_report["variants"].items()
    }
    training_report_path.write_text(
        json.dumps(training_report, indent=2) + "\n", encoding="utf-8"
    )
    json_log(
        completed=True,
        validation_selected_candidate=validation_choice,
        final_test={
            variant: compact_metrics({"metrics": row["metrics"]})
            for variant, row in test_report["variants"].items()
        },
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
