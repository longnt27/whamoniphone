#!/usr/bin/env python3
"""Downstream-aware FastViT fine-tuning through a frozen WHAM core.

This is deliberately separate from the 3DPW test evaluator. It trains only on
the official parsed 3DPW training split, selects checkpoints on the validation
split, and leaves the test split untouched for the final A/B decision.

The official parsed training file stores numeric track ids. Raw 3DPW
``sequenceFiles/train`` annotations are therefore used to recover the matching
image sequence by comparing camera poses. No test annotations enter training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_wham_feature_substitution as wham_ab
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REQUIRED_TRAIN_KEYS = {
    "bbox",
    "betas",
    "res",
    "vid",
    "pose",
    "kp2d",
    "joints3D",
    "frame_id",
    "cam_poses",
    "features",
    "gender",
}
REQUIRED_VAL_KEYS = {
    "bbox",
    "res",
    "vid",
    "pose",
    "kp2d",
    "init_kp3d",
    "init_pose",
    "frame_id",
    "features",
}


@dataclass(frozen=True)
class TrainTrack:
    track_id: int
    sequence: str
    indices: np.ndarray
    camera_error: float


@dataclass(frozen=True)
class RawSequence:
    cameras: np.ndarray
    betas: tuple[np.ndarray, ...]
    genders: tuple[int, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_bbox(values: Any) -> np.ndarray:
    bbox = wham_ab.to_numpy(values).astype(np.float32)
    return bbox[..., [0, 1, -1]]


def take_rows(values: Any, indices: np.ndarray) -> Any:
    if isinstance(values, torch.Tensor):
        return values[torch.from_numpy(indices.astype(np.int64))]
    return np.asarray(values)[indices]


def contiguous_track_indices(video_ids: Any) -> list[np.ndarray]:
    values = wham_ab.to_numpy(video_ids).reshape(-1)
    if not len(values):
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    groups = np.split(np.arange(len(values), dtype=np.int64), boundaries)
    seen: set[int] = set()
    for group in groups:
        track_id = int(values[group[0]])
        if track_id in seen:
            raise RuntimeError(f"Training track id {track_id} is not contiguous")
        seen.add(track_id)
    return groups


def locate_sequence_train_root(root: Path) -> Path:
    candidates = (
        root / "sequenceFiles" / "train",
        root / "sequenceFiles" / "sequenceFiles" / "train",
        root / "3DPW" / "sequenceFiles" / "train",
        root / "train",
    )
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("*.pkl"), None) is not None:
            return candidate
    raise FileNotFoundError(
        "Could not find licensed 3DPW sequenceFiles/train annotations below "
        f"{root}"
    )


def load_sequence_catalog(sequence_root: Path) -> dict[str, RawSequence]:
    catalog: dict[str, RawSequence] = {}
    for path in sorted(sequence_root.glob("*.pkl")):
        with path.open("rb") as stream:
            payload = pickle.load(stream, encoding="latin1")
        cameras = np.asarray(payload["cam_poses"], dtype=np.float32)
        if cameras.ndim != 3 or cameras.shape[-2:] != (4, 4):
            raise RuntimeError(f"Unexpected cam_poses shape in {path}: {cameras.shape}")
        betas = tuple(
            np.asarray(values, dtype=np.float32)[:10]
            for values in payload["betas"]
        )
        gender_codes: list[int] = []
        for value in payload["genders"]:
            if isinstance(value, bytes):
                value = value.decode("ascii")
            normalized = str(value).lower()
            gender_codes.append({"m": 0, "male": 0, "f": 1, "female": 1}[normalized])
        genders = tuple(gender_codes)
        if len(betas) != len(genders):
            raise RuntimeError(f"Person metadata mismatch in {path}")
        catalog[path.stem] = RawSequence(cameras, betas, genders)
    if not catalog:
        raise RuntimeError(f"No sequence annotation files found in {sequence_root}")
    return catalog


def map_training_tracks(
    labels: dict[str, Any], sequence_root: Path
) -> list[TrainTrack]:
    """Recover numeric parsed track ids as raw image sequence names."""

    catalog = load_sequence_catalog(sequence_root)
    video_ids = wham_ab.to_numpy(labels["vid"]).reshape(-1)
    parsed_cameras = wham_ab.to_numpy(labels["cam_poses"]).astype(np.float32)
    parsed_frame_ids = wham_ab.to_numpy(labels["frame_id"]).astype(np.int64)
    parsed_betas = wham_ab.to_numpy(labels["betas"]).astype(np.float32)
    parsed_genders = wham_ab.to_numpy(labels["gender"]).astype(np.int64)
    tracks: list[TrainTrack] = []
    for group in contiguous_track_indices(video_ids):
        track_id = int(video_ids[group[0]])
        probe_positions = np.unique(
            np.linspace(0, len(group) - 1, num=min(16, len(group)), dtype=np.int64)
        )
        probe_rows = group[probe_positions]
        frame_ids = parsed_frame_ids[probe_rows]
        expected = parsed_cameras[probe_rows]
        expected_beta = parsed_betas[group[0], :10]
        expected_gender = int(parsed_genders[group[0]])
        errors: list[tuple[float, str]] = []
        for sequence, raw in catalog.items():
            identity_matches = any(
                gender == expected_gender
                and np.max(np.abs(beta - expected_beta)) <= 1e-5
                for beta, gender in zip(raw.betas, raw.genders)
            )
            if not identity_matches or int(frame_ids.max()) >= len(raw.cameras):
                continue
            observed = raw.cameras[frame_ids]
            error = float(np.max(np.abs(observed - expected)))
            errors.append((error, sequence))
        if not errors:
            raise RuntimeError(f"No raw camera candidate for training track {track_id}")
        errors.sort()
        best_error, sequence = errors[0]
        if best_error > 1e-5:
            preview = ", ".join(f"{name}={error:.3g}" for error, name in errors[:3])
            raise RuntimeError(
                f"Could not map training track {track_id} by camera pose; {preview}"
            )
        equally_exact = [name for error, name in errors if abs(error - best_error) < 1e-8]
        if len(equally_exact) > 1:
            raise RuntimeError(
                f"Ambiguous camera-pose mapping for training track {track_id}: "
                + ", ".join(equally_exact)
            )
        tracks.append(TrainTrack(track_id, sequence, group, best_error))
    return tracks


def sequence_paths(
    image_root: Path, sequence: str, frame_ids: np.ndarray
) -> list[Path]:
    sequence_dir = image_root / sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Missing raw image sequence: {sequence_dir}")
    all_images: list[Path] | None = None
    paths: list[Path] = []
    for raw_frame_id in frame_ids:
        frame_id = int(raw_frame_id)
        direct = sequence_dir / f"image_{frame_id:05d}.jpg"
        if direct.is_file():
            paths.append(direct)
            continue
        if all_images is None:
            all_images = sorted(sequence_dir.glob("image_*.jpg"))
        if 0 <= frame_id < len(all_images):
            paths.append(all_images[frame_id])
            continue
        raise FileNotFoundError(f"Missing frame {frame_id} in {sequence_dir}")
    return paths


class TemporalClipDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        labels: dict[str, Any],
        tracks: list[TrainTrack],
        image_root: Path,
        wham_repo: Path,
        clip_length: int,
        stride: int,
        max_clips: int,
        seed: int,
    ) -> None:
        self.labels = labels
        self.tracks = tracks
        self.image_root = image_root
        self.wham_repo = wham_repo
        self.clip_length = clip_length
        self._process_image: Any = None
        clips: list[tuple[int, int]] = []
        for track_index, track in enumerate(tracks):
            maximum_start = len(track.indices) - (clip_length + 1)
            for start in range(0, maximum_start + 1, stride):
                clips.append((track_index, start))
        random.Random(seed).shuffle(clips)
        if max_clips > 0:
            clips = clips[:max_clips]
        if not clips:
            raise RuntimeError(
                f"No training clips of {clip_length + 1} frames were available"
            )
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def process_image(self) -> Any:
        if self._process_image is None:
            self._process_image = wham_ab.load_process_image(self.wham_repo)
        return self._process_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index, start = self.clips[index]
        track = self.tracks[track_index]
        rows = track.indices[start : start + self.clip_length + 1]
        target_rows = rows[1:]
        frame_ids = wham_ab.to_numpy(
            take_rows(self.labels["frame_id"], target_rows)
        ).astype(np.int64)
        boxes_all = compact_bbox(take_rows(self.labels["bbox"], rows))
        boxes = boxes_all[1:]
        paths = sequence_paths(self.image_root, track.sequence, frame_ids)
        images: list[torch.Tensor] = []
        process_image = self.process_image()
        for path, box in zip(paths, boxes):
            with wham_ab.Image.open(path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            normalized, _ = process_image(
                rgb, box[:2], float(box[2]) / 200.0, 256, 256
            )
            images.append(
                torch.from_numpy(np.asarray(normalized, dtype=np.float32))
            )

        keypoints = wham_ab.to_numpy(take_rows(self.labels["kp2d"], rows))
        resolution = wham_ab.to_numpy(take_rows(self.labels["res"], rows[:1]))[0]
        x_all, _ = wham_ab.normalized_wham_input(
            keypoints, boxes_all, resolution
        )
        # Match Dataset3D.get_inputs: training detections use vis_thr=0.6.
        mask_all = keypoints[..., 2] < 0.6
        initial_joints = wham_ab.root_center_coco(
            wham_ab.to_numpy(take_rows(self.labels["joints3D"], rows[:1]))[0]
        )
        initial_keypoints = np.concatenate(
            (initial_joints.reshape(-1), x_all[0]), axis=0
        ).astype(np.float32)
        initial_axis_angle = torch.from_numpy(
            wham_ab.to_numpy(take_rows(self.labels["pose"], rows[:1]))[0]
            .astype(np.float32)
            .reshape(24, 3)
        )
        initial_pose = wham_ab.matrix_to_rotation_6d(
            wham_ab.axis_angle_to_matrix(initial_axis_angle)
        )
        teacher_features = torch.from_numpy(
            wham_ab.to_numpy(take_rows(self.labels["features"], target_rows))
            .astype(np.float32)
        )
        target_axis_angle = torch.from_numpy(
            wham_ab.to_numpy(take_rows(self.labels["pose"], target_rows))
            .astype(np.float32)
            .reshape(self.clip_length, 24, 3)
        )
        target_pose = wham_ab.matrix_to_rotation_6d(
            wham_ab.axis_angle_to_matrix(target_axis_angle)
        )
        return {
            "images": torch.stack(images),
            "x": torch.from_numpy(x_all[1:]),
            "mask": torch.from_numpy(mask_all[1:]),
            "init_kp": torch.from_numpy(initial_keypoints).reshape(1, 88),
            "init_pose": initial_pose.reshape(1, 24, 6),
            "teacher_features": teacher_features,
            "target_pose": target_pose,
        }


def configure_stage(model: wham_ab.FastViTHMR2Student, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.spatial_head.parameters():
        parameter.requires_grad_(True)
    if stage == "downstream_last_stage":
        for parameter in model.backbone.stages[3].parameters():
            parameter.requires_grad_(True)
        for parameter in model.backbone.final_conv.parameters():
            parameter.requires_grad_(True)


def set_training_modes(model: wham_ab.FastViTHMR2Student, stage: str) -> None:
    model.eval()
    model.spatial_head.train()
    if stage == "downstream_last_stage":
        model.backbone.stages[3].train()
        model.backbone.final_conv.train()


def enable_frozen_wham_input_gradients(network: nn.Module) -> None:
    """Allow cuDNN LSTM input gradients without training WHAM parameters.

    cuDNN does not retain the reserve buffer needed for an RNN backward pass
    when the RNN is in evaluation mode. Only the motion-decoder LSTM lies on
    the student-feature gradient path. Put that LSTM in training mode while
    forcing its internal inter-layer dropout to zero, which matches evaluation
    behavior and keeps every WHAM parameter frozen.
    """

    network.eval()
    decoder_rnn = network.motion_decoder.regressor.rnn
    decoder_rnn.dropout = 0.0
    decoder_rnn.train()


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def frozen_motion_context(
    network: nn.Module, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    processed = network.preprocess(batch["x"].clone(), batch["mask"])
    _, context = network.motion_encoder(processed, batch["init_kp"])
    return context


def decode_pose(
    network: nn.Module,
    context: torch.Tensor,
    features: torch.Tensor,
    initial_pose: torch.Tensor,
) -> torch.Tensor:
    integrated = network.integrator(context, features)
    pose, _, _, _ = network.motion_decoder(integrated, initial_pose)
    return pose.reshape(*pose.shape[:2], 24, 6)


def rotation_fidelity_loss(
    prediction_6d: torch.Tensor,
    target_6d: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    # Rotation geometry is numerically fragile in float16, so keep this small
    # loss calculation in float32 even when the enclosing forward uses AMP.
    prediction = wham_ab.rotation_6d_to_matrix(prediction_6d.float())
    target = wham_ab.rotation_6d_to_matrix(target_6d.float())
    relative = prediction @ target.transpose(-1, -2)
    cosine = (
        relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0
    ).div(2.0).clamp(-1.0, 1.0)
    if mode == "cosine":
        return 1.0 - cosine
    if mode == "geodesic":
        # ||vee(R - R^T)|| / 2 = |sin(theta)|. atan2 avoids the
        # near-zero derivative collapse of 1-cos(theta), and returns the same
        # angular quantity used by the validation gates, in radians.
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
    raise ValueError(f"Unsupported rotation loss: {mode}")


def downstream_losses(
    model: wham_ab.FastViTHMR2Student,
    network: nn.Module,
    batch: dict[str, torch.Tensor],
    student_features: torch.Tensor,
    weights: dict[str, float],
    rotation_loss_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        context = frozen_motion_context(network, batch)
        teacher_pose = decode_pose(
            network,
            context,
            batch["teacher_features"],
            batch["init_pose"],
        )
    student_pose = decode_pose(
        network, context.detach(), student_features, batch["init_pose"]
    )
    prediction_z = (student_features - model.target_mean) / model.target_std
    target_z = (batch["teacher_features"] - model.target_mean) / model.target_std
    teacher_rotation_loss = rotation_fidelity_loss(
        student_pose, teacher_pose, rotation_loss_mode
    )
    ground_truth_rotation_loss = rotation_fidelity_loss(
        student_pose, batch["target_pose"], rotation_loss_mode
    )
    losses = {
        "token": F.mse_loss(prediction_z, target_z),
        "raw_cosine": (
            1.0
            - F.cosine_similarity(
                student_features, batch["teacher_features"], dim=-1
            )
        ).mean(),
        "centered_cosine": (
            1.0 - F.cosine_similarity(prediction_z, target_z, dim=-1, eps=1e-6)
        ).mean(),
        "wham_pose": teacher_rotation_loss.mean(),
        "wham_root": teacher_rotation_loss[..., 0].mean(),
        "wham_gt_pose": ground_truth_rotation_loss.mean(),
        "wham_gt_root": ground_truth_rotation_loss[..., 0].mean(),
    }
    total = sum(weights[name] * value for name, value in losses.items())
    return total, losses


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def gradient_preflight(
    model: wham_ab.FastViTHMR2Student,
    network: nn.Module,
    dataset: TemporalClipDataset,
    device: torch.device,
    weights: dict[str, float],
    rotation_loss_mode: str,
) -> dict[str, float]:
    """Exercise one real CUDA backward pass without updating either model."""

    configure_stage(model, "downstream_head")
    set_training_modes(model, "downstream_head")
    enable_frozen_wham_input_gradients(network)
    batch = move_batch(
        {key: value.unsqueeze(0) for key, value in dataset[0].items()}, device
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        frames = batch["images"].shape[1]
        features = model(batch["images"].reshape(-1, 3, 256, 256)).reshape(
            1, frames, 1024
        )
        loss, _ = downstream_losses(
            model, network, batch, features, weights, rotation_loss_mode
        )
    loss.backward()
    gradients = [
        parameter.grad.float().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Gradient preflight produced no FastViT gradients")
    grad_norm = torch.linalg.vector_norm(torch.cat(gradients))
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(
            f"Gradient preflight was non-finite: loss={loss}, norm={grad_norm}"
        )
    if any(parameter.grad is not None for parameter in network.parameters()):
        raise RuntimeError("Frozen WHAM unexpectedly received parameter gradients")
    result = {"loss": float(loss.detach()), "fastvit_gradient_norm": float(grad_norm)}
    model.zero_grad(set_to_none=True)
    model.eval()
    network.eval()
    return result


def train_epoch(
    model: wham_ab.FastViTHMR2Student,
    network: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    stage: str,
    weights: dict[str, float],
    rotation_loss_mode: str,
    log_every: int,
) -> dict[str, float]:
    set_training_modes(model, stage)
    enable_frozen_wham_input_gradients(network)
    totals = {"loss": 0.0, **{name: 0.0 for name in weights}}
    samples = 0
    started = time.perf_counter()
    for step, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        batch_size, frames = batch["images"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            flat_images = batch["images"].reshape(-1, 3, 256, 256)
            student_features = model(flat_images).reshape(batch_size, frames, 1024)
            loss, components = downstream_losses(
                model,
                network,
                batch,
                student_features,
                weights,
                rotation_loss_mode,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        samples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value.detach()) * batch_size
        if step % log_every == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "step": step,
                        "steps": len(loader),
                        "loss": totals["loss"] / samples,
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def validate_downstream(
    model: wham_ab.FastViTHMR2Student,
    network: nn.Module,
    labels: dict[str, Any],
    image_root: Path,
    process_image: Any,
    device: torch.device,
    max_tracks: int,
    max_frames: int,
    feature_batch_size: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    network.eval()
    matching = [
        index
        for index, raw_video_id in enumerate(labels["vid"])
        if (image_root / str(raw_video_id).rsplit("_", 1)[0]).is_dir()
    ]
    if not matching:
        raise RuntimeError("No validation track names matched raw 3DPW images")
    if max_tracks > 0 and len(matching) > max_tracks:
        positions = np.linspace(
            0, len(matching) - 1, num=max_tracks, dtype=np.int64
        )
        matching = [matching[int(position)] for position in positions]

    all_teacher_features: list[torch.Tensor] = []
    all_student_features: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_teacher_pose: list[torch.Tensor] = []
    all_student_pose: list[torch.Tensor] = []
    per_track: list[dict[str, Any]] = []
    for index in matching:
        video_id = str(labels["vid"][index])
        keypoints = wham_ab.to_numpy(labels["kp2d"][index])
        bbox = compact_bbox(labels["bbox"][index])
        frame_ids = wham_ab.to_numpy(labels["frame_id"][index])
        resolution = wham_ab.to_numpy(labels["res"][index][0])
        available = min(
            max_frames if max_frames > 0 else len(keypoints) - 1,
            len(keypoints) - 1,
            len(bbox) - 1,
            len(frame_ids) - 1,
            len(labels["features"][index]) - 1,
            len(labels["pose"][index]) - 1,
        )
        if available <= 0:
            continue
        frame_slice = slice(1, available + 1)
        paths = wham_ab.sequence_image_paths(
            image_root, video_id, frame_ids[frame_slice]
        )
        student_features, _ = wham_ab.extract_student_features(
            model,
            paths,
            bbox[frame_slice],
            process_image,
            feature_batch_size,
            device,
        )
        teacher_features = torch.from_numpy(
            wham_ab.to_numpy(labels["features"][index])[frame_slice].astype(
                np.float32
            )
        )
        x_all, mask_all = wham_ab.normalized_wham_input(
            keypoints, bbox, resolution
        )
        initial_joints = wham_ab.root_center_coco(
            wham_ab.to_numpy(labels["init_kp3d"][index])[0]
        )
        initial_keypoints = np.concatenate(
            (initial_joints.reshape(-1), x_all[0]), axis=0
        ).astype(np.float32)
        initial_axis_angle = torch.from_numpy(
            wham_ab.to_numpy(labels["init_pose"][index])[0]
            .astype(np.float32)
            .reshape(24, 3)
        )
        initial_pose = wham_ab.matrix_to_rotation_6d(
            wham_ab.axis_angle_to_matrix(initial_axis_angle)
        ).reshape(1, 1, 24, 6)
        target = torch.from_numpy(
            wham_ab.to_numpy(labels["pose"][index])[frame_slice].astype(
                np.float32
            )
        ).reshape(available, 24, 3)
        common = {
            "network": network,
            "x": torch.from_numpy(x_all[frame_slice]).unsqueeze(0).to(device),
            "mask": torch.from_numpy(mask_all[frame_slice])
            .unsqueeze(0)
            .to(device),
            "init_kp": torch.from_numpy(initial_keypoints)
            .reshape(1, 1, 88)
            .to(device),
            "init_pose": initial_pose.to(device),
            "init_root": wham_ab.matrix_to_rotation_6d(
                wham_ab.axis_angle_to_matrix(target[0, 0])
            )
            .reshape(1, 1, 6)
            .to(device),
            "cam_angvel": torch.zeros(1, available, 6, device=device),
        }
        teacher_output = wham_ab.run_wham_core(
            features=teacher_features.unsqueeze(0).to(device), **common
        )
        student_output = wham_ab.run_wham_core(
            features=student_features.unsqueeze(0).to(device), **common
        )
        teacher_pose = teacher_output["pose"].squeeze(0).cpu()
        student_pose = student_output["pose"].squeeze(0).cpu()
        teacher_error = wham_ab.pose_error_against_axis_angle(
            teacher_pose, target
        )
        student_error = wham_ab.pose_error_against_axis_angle(
            student_pose, target
        )
        drift = wham_ab.pose_drift(student_pose, teacher_pose)
        per_track.append(
            {
                "sequence": video_id,
                "frames": available,
                "feature_cosine": float(
                    F.cosine_similarity(
                        student_features, teacher_features, dim=-1
                    ).mean()
                ),
                "teacher_pose_error_deg": float(teacher_error.mean()),
                "student_pose_error_deg": float(student_error.mean()),
                "pose_degradation_deg": float(
                    student_error.mean() - teacher_error.mean()
                ),
                "pose_drift_deg": float(drift.mean()),
            }
        )
        all_teacher_features.append(teacher_features)
        all_student_features.append(student_features)
        all_targets.append(target)
        all_teacher_pose.append(teacher_pose)
        all_student_pose.append(student_pose)
    if not per_track:
        raise RuntimeError("No non-empty validation tracks were evaluated")

    teacher_features = torch.cat(all_teacher_features)
    student_features = torch.cat(all_student_features)
    target = torch.cat(all_targets)
    teacher_pose = torch.cat(all_teacher_pose)
    student_pose = torch.cat(all_student_pose)
    teacher_error = wham_ab.pose_error_against_axis_angle(
        teacher_pose, target
    ).numpy()
    student_error = wham_ab.pose_error_against_axis_angle(
        student_pose, target
    ).numpy()
    drift = wham_ab.pose_drift(student_pose, teacher_pose).numpy()
    feature_cosine = F.cosine_similarity(
        student_features, teacher_features, dim=-1
    ).numpy()
    teacher_mean = float(teacher_error.mean())
    student_mean = float(student_error.mean())
    degradation = student_mean - teacher_mean
    relative = degradation / max(teacher_mean, 1e-8)
    drift_mean = float(drift.mean())
    gates = {
        "absolute_pose_degradation": degradation
        <= thresholds["max_pose_degradation_deg"],
        "relative_pose_degradation": relative
        <= thresholds["max_relative_pose_degradation"],
        "student_teacher_pose_drift": drift_mean
        <= thresholds["max_teacher_drift_deg"],
    }
    return {
        "accepted_on_validation": all(gates.values()),
        "gates": gates,
        "tracks": len(per_track),
        "frames": len(target),
        "feature_cosine_mean": float(feature_cosine.mean()),
        "feature_cosine_p05": float(np.percentile(feature_cosine, 5)),
        "feature_rmse": float(
            torch.sqrt(F.mse_loss(student_features, teacher_features))
        ),
        "teacher_pose_error_deg": teacher_mean,
        "student_pose_error_deg": student_mean,
        "pose_degradation_deg": degradation,
        "relative_pose_degradation": relative,
        "student_teacher_pose_drift_deg": drift_mean,
        "student_teacher_pose_drift_p95_deg": float(np.percentile(drift, 95)),
        "per_track": per_track,
    }


def validation_score(metrics: dict[str, Any], thresholds: dict[str, float]) -> float:
    ratios = (
        metrics["pose_degradation_deg"]
        / thresholds["max_pose_degradation_deg"],
        metrics["relative_pose_degradation"]
        / thresholds["max_relative_pose_degradation"],
        metrics["student_teacher_pose_drift_deg"]
        / thresholds["max_teacher_drift_deg"],
    )
    return float(max(ratios))


def compact_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_on_validation": metrics["accepted_on_validation"],
        "gates": metrics["gates"],
        "tracks": metrics["tracks"],
        "frames": metrics["frames"],
        "feature_cosine_mean": metrics["feature_cosine_mean"],
        "teacher_pose_error_deg": metrics["teacher_pose_error_deg"],
        "student_pose_error_deg": metrics["student_pose_error_deg"],
        "pose_degradation_deg": metrics["pose_degradation_deg"],
        "relative_pose_degradation": metrics["relative_pose_degradation"],
        "student_teacher_pose_drift_deg": metrics[
            "student_teacher_pose_drift_deg"
        ],
    }


def validation_acceptance_state(validation: dict[str, Any]) -> str:
    if validation["accepted_on_validation"]:
        return "awaiting_untouched_3dpw_test"
    return "rejected_on_validation"


def checkpoint_payload(
    source: dict[str, Any],
    model: wham_ab.FastViTHMR2Student,
    epoch: int,
    stage: str,
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(source)
    payload.update(
        {
            "student_state_dict": model.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "accepted": False,
            "acceptance_state": validation_acceptance_state(validation),
            "phase3_validation": validation,
            "phase3_metadata": metadata,
        }
    )
    return payload


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    missing_train = sorted(REQUIRED_TRAIN_KEYS - set(train_labels))
    missing_val = sorted(REQUIRED_VAL_KEYS - set(val_labels))
    if missing_train or missing_val:
        raise RuntimeError(
            f"Missing parsed keys: train={missing_train}, validation={missing_val}"
        )
    image_root = wham_ab.locate_image_root(args.three_dpw_root)
    sequence_root = locate_sequence_train_root(
        args.sequence_root or args.three_dpw_root
    )
    tracks = map_training_tracks(train_labels, sequence_root)
    dataset = TemporalClipDataset(
        train_labels,
        tracks,
        image_root,
        args.wham_repo,
        args.clip_length,
        args.stride,
        args.max_clips,
        args.seed,
    )
    sample = dataset[0]
    expected_shapes = {
        "images": (args.clip_length, 3, 256, 256),
        "x": (args.clip_length, 37),
        "mask": (args.clip_length, 17),
        "init_kp": (1, 88),
        "init_pose": (1, 24, 6),
        "teacher_features": (args.clip_length, 1024),
        "target_pose": (args.clip_length, 24, 6),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in sample.items()}
    mismatches = {
        key: {"expected": shape, "actual": actual_shapes.get(key)}
        for key, shape in expected_shapes.items()
        if actual_shapes.get(key) != shape
    }
    if mismatches:
        raise RuntimeError(f"Unexpected training sample shapes: {mismatches}")
    sample_paths = sequence_paths(
        image_root,
        tracks[0].sequence,
        wham_ab.to_numpy(
            take_rows(train_labels["frame_id"], tracks[0].indices[:3])
        ),
    )
    val_matches = sum(
        (image_root / str(video_id).rsplit("_", 1)[0]).is_dir()
        for video_id in val_labels["vid"]
    )
    if val_matches == 0:
        raise RuntimeError("No validation labels matched raw image sequence names")
    return {
        "image_root": str(image_root),
        "sequence_train_root": str(sequence_root),
        "training_rows": len(wham_ab.to_numpy(train_labels["vid"])),
        "training_tracks": len(tracks),
        "training_clips": len(dataset),
        "validation_tracks_matched": int(val_matches),
        "first_track": tracks[0].sequence,
        "sample_images": [str(path) for path in sample_paths],
        "sample_shapes": actual_shapes,
        "max_camera_mapping_error": max(track.camera_error for track in tracks),
    }


def self_test() -> None:
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3)
    target = wham_ab.matrix_to_rotation_6d(identity)
    for mode in ("cosine", "geodesic"):
        loss = rotation_fidelity_loss(target, target, mode)
        assert float(loss.max()) < 1e-7
        perturbed = target.clone()
        perturbed[..., 1] += 0.1
        perturbed.requires_grad_(True)
        rotation_fidelity_loss(perturbed, target, mode).mean().backward()
        assert perturbed.grad is not None
    groups = contiguous_track_indices(torch.tensor([0, 0, 1, 1, 1, 2]))
    assert [len(group) for group in groups] == [2, 3, 1]
    print("Self-test passed: track grouping and differentiable WHAM pose loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        help=(
            "Optional separate root containing licensed sequenceFiles/train; "
            "defaults to --three-dpw-root"
        ),
    )
    parser.add_argument("--train-parsed", type=Path, required=True)
    parser.add_argument("--val-parsed", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--last-stage-epochs", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--finetune-head-lr", type=float, default=2e-5)
    parser.add_argument("--finetune-backbone-lr", type=float, default=2e-6)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--raw-cosine-weight", type=float, default=0.25)
    parser.add_argument("--centered-cosine-weight", type=float, default=0.25)
    parser.add_argument("--wham-pose-weight", type=float, default=25.0)
    parser.add_argument("--wham-root-weight", type=float, default=10.0)
    parser.add_argument("--wham-gt-pose-weight", type=float, default=0.0)
    parser.add_argument("--wham-gt-root-weight", type=float, default=0.0)
    parser.add_argument(
        "--rotation-loss",
        choices=("cosine", "geodesic"),
        default="cosine",
        help="Rotation-space loss used for frozen-WHAM supervision",
    )
    parser.add_argument("--val-tracks", type=int, default=0)
    parser.add_argument("--val-frames", type=int, default=300)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--max-pose-degradation-deg", type=float, default=1.0)
    parser.add_argument("--max-relative-pose-degradation", type=float, default=0.10)
    parser.add_argument("--max-teacher-drift-deg", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--stop-when-accepted",
        action="store_true",
        help="Stop after the first validation checkpoint that satisfies every gate",
    )
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required_files = (
        args.train_parsed,
        args.val_parsed,
        args.source_checkpoint,
        args.wham_checkpoint,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if wham_commit != wham_ab.WHAM_COMMIT:
        raise RuntimeError(
            f"Expected WHAM commit {wham_ab.WHAM_COMMIT}, found {wham_commit}"
        )
    preflight = inspect_inputs(args)
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.inspect_data:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for downstream fine-tuning")

    seed_everything(args.seed)
    device = torch.device("cuda")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    image_root = wham_ab.locate_image_root(args.three_dpw_root)
    sequence_root = locate_sequence_train_root(
        args.sequence_root or args.three_dpw_root
    )
    tracks = map_training_tracks(train_labels, sequence_root)
    train_dataset = TemporalClipDataset(
        train_labels,
        tracks,
        image_root,
        args.wham_repo,
        args.clip_length,
        args.stride,
        args.max_clips,
        args.seed,
    )
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(train_dataset, **loader_options)
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    model, source_training = wham_ab.load_student(args.source_checkpoint, device)
    network = wham_ab.load_wham_core(
        args.wham_repo.resolve(), args.wham_checkpoint, device
    )
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    network.eval()
    process_image = wham_ab.load_process_image(args.wham_repo.resolve())
    thresholds = {
        "max_pose_degradation_deg": args.max_pose_degradation_deg,
        "max_relative_pose_degradation": args.max_relative_pose_degradation,
        "max_teacher_drift_deg": args.max_teacher_drift_deg,
    }
    weights = {
        "token": args.token_weight,
        "raw_cosine": args.raw_cosine_weight,
        "centered_cosine": args.centered_cosine_weight,
        "wham_pose": args.wham_pose_weight,
        "wham_root": args.wham_root_weight,
        "wham_gt_pose": args.wham_gt_pose_weight,
        "wham_gt_root": args.wham_gt_root_weight,
    }
    metadata = {
        "schema_version": 1,
        "method": (
            "frozen_wham_supervised_pose_and_feature_distillation"
            if args.wham_gt_pose_weight > 0 or args.wham_gt_root_weight > 0
            else "frozen_wham_temporal_feature_distillation"
        ),
        "source_checkpoint_sha256": wham_ab.sha256_file(args.source_checkpoint),
        "source_training": source_training,
        "wham_commit": wham_ab.WHAM_COMMIT,
        "wham_checkpoint_sha256": wham_ab.sha256_file(args.wham_checkpoint),
        "train_parsed_sha256": wham_ab.sha256_file(args.train_parsed),
        "val_parsed_sha256": wham_ab.sha256_file(args.val_parsed),
        "test_data_used": False,
        "clip_length": args.clip_length,
        "stride": args.stride,
        "training_tracks": len(tracks),
        "training_clips": len(train_dataset),
        "loss_weights": weights,
        "rotation_loss": args.rotation_loss,
        "thresholds": thresholds,
        "seed": args.seed,
    }

    preflight_gradient = gradient_preflight(
        model, network, train_dataset, device, weights, args.rotation_loss
    )
    print(json.dumps({"gradient_preflight": preflight_gradient}, indent=2), flush=True)

    print("Evaluating the unchanged phase-two checkpoint on 3DPW validation...")
    baseline = validate_downstream(
        model,
        network,
        val_labels,
        image_root,
        process_image,
        device,
        args.val_tracks,
        args.val_frames,
        args.feature_batch_size,
        thresholds,
    )
    print(
        json.dumps({"baseline_validation": compact_validation(baseline)}, indent=2),
        flush=True,
    )
    best_validation = baseline
    best_score = validation_score(baseline, thresholds)
    best_path = work_dir / "fastvit_hmr2_best.pth"
    source_epoch = int(source_checkpoint.get("epoch", 0))
    best_epoch = source_epoch
    best_stage = "downstream_baseline"
    torch.save(
        checkpoint_payload(
            source_checkpoint,
            model,
            source_epoch,
            "downstream_baseline",
            baseline,
            metadata,
        ),
        best_path,
    )
    history: list[dict[str, Any]] = []
    stages = (
        ("downstream_head", args.head_epochs),
        ("downstream_last_stage", args.last_stage_epochs),
    )
    phase_epoch = 0
    stop_training = False
    for stage, epochs in stages:
        if epochs <= 0:
            continue
        configure_stage(model, stage)
        head_parameters = [
            parameter
            for parameter in model.spatial_head.parameters()
            if parameter.requires_grad
        ]
        if stage == "downstream_head":
            groups = [{"params": head_parameters, "lr": args.head_lr}]
        else:
            backbone_parameters = [
                parameter
                for parameter in model.backbone.parameters()
                if parameter.requires_grad
            ]
            groups = [
                {"params": head_parameters, "lr": args.finetune_head_lr},
                {
                    "params": backbone_parameters,
                    "lr": args.finetune_backbone_lr,
                },
            ]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.02)
        total_steps = max(len(loader) * epochs, 1)
        warmup_steps = min(len(loader), max(total_steps // 10, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_steps, warmup=warmup_steps: cosine_schedule(
                step, total, warmup
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for _ in range(epochs):
            phase_epoch += 1
            started = time.perf_counter()
            training = train_epoch(
                model,
                network,
                loader,
                optimizer,
                scheduler,
                scaler,
                device,
                stage,
                weights,
                args.rotation_loss,
                args.log_every,
            )
            validation = validate_downstream(
                model,
                network,
                val_labels,
                image_root,
                process_image,
                device,
                args.val_tracks,
                args.val_frames,
                args.feature_batch_size,
                thresholds,
            )
            score = validation_score(validation, thresholds)
            row = {
                "epoch": source_epoch + phase_epoch,
                "stage": stage,
                **{f"train_{key}": value for key, value in training.items()},
                "val_feature_cosine_mean": validation["feature_cosine_mean"],
                "val_teacher_pose_error_deg": validation[
                    "teacher_pose_error_deg"
                ],
                "val_student_pose_error_deg": validation[
                    "student_pose_error_deg"
                ],
                "val_pose_degradation_deg": validation[
                    "pose_degradation_deg"
                ],
                "val_relative_pose_degradation": validation[
                    "relative_pose_degradation"
                ],
                "val_pose_drift_deg": validation[
                    "student_teacher_pose_drift_deg"
                ],
                "val_accepted": validation["accepted_on_validation"],
                "selection_score": score,
                "minutes": (time.perf_counter() - started) / 60.0,
            }
            history.append(row)
            print(json.dumps(row, indent=2), flush=True)
            if score < best_score:
                best_score = score
                best_validation = validation
                best_epoch = source_epoch + phase_epoch
                best_stage = stage
                torch.save(
                    checkpoint_payload(
                        source_checkpoint,
                        model,
                        source_epoch + phase_epoch,
                        stage,
                        validation,
                        metadata,
                    ),
                    best_path,
                )
            write_history(work_dir / "fastvit_hmr2_history.csv", history)
            if validation["accepted_on_validation"] and args.stop_when_accepted:
                print("All validation gates passed; stopping this continuation early.")
                stop_training = True
                break
        if stop_training:
            break

    report = {
        "schema_version": 1,
        "accepted_on_validation": best_validation["accepted_on_validation"],
        "deployment_accepted": False,
        "acceptance_state": validation_acceptance_state(best_validation),
        "baseline_validation": baseline,
        "best_validation": best_validation,
        "best_selection_score": best_score,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": wham_ab.sha256_file(best_path),
        "metadata": metadata,
    }
    report_path = work_dir / "fastvit_hmr2_training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted_on_validation": report["accepted_on_validation"],
                "deployment_accepted": report["deployment_accepted"],
                "acceptance_state": report["acceptance_state"],
                "best_validation": compact_validation(best_validation),
                "best_checkpoint_sha256": report["best_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Selected validation checkpoint: {best_path}")
    if best_validation["accepted_on_validation"]:
        print("Deployment remains blocked until the untouched 3DPW test A/B passes.")
    else:
        print("Validation rejected this checkpoint; do not run the 3DPW test A/B.")


if __name__ == "__main__":
    main()
