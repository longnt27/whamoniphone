#!/usr/bin/env python3
"""Deployment-aware training for the compact iPhone WHAM pipeline.

The source FastViT was distilled and downstream-tuned with official crops and
keypoints.  This program closes the remaining train/deploy mismatch by using
YOLO26 observations for every training clip, learning compact first-frame pose
and 3D-joint heads, and adapting only WHAM's input-facing layers.  Model
selection uses 3DPW validation; the test split is deliberately not accepted as
an argument here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as feature_eval
import finetune_fastvit_wham_downstream as phase3
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

DEPLOYMENT_SCHEMA = "yolo26_fastvit_wham_split_v1"
YOLO26_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
SOURCE_STUDENT_SHA256 = (
    "f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1"
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


def canonical_rotation_6d(raw: torch.Tensor) -> torch.Tensor:
    matrix = feature_eval.rotation_6d_to_matrix(raw.reshape(*raw.shape[:-1], 24, 6))
    return feature_eval.matrix_to_rotation_6d(matrix)


class DeploymentInitializer(nn.Module):
    """Small, Core-ML-friendly pose and 3D-joint heads on the FastViT token."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),
            nn.GELU(),
        )
        self.pose = nn.Linear(hidden_dim, 24 * 6)
        self.joints = nn.Linear(hidden_dim, 17 * 3)
        nn.init.xavier_uniform_(self.pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.joints.weight, gain=0.01)
        identity = feature_eval.matrix_to_rotation_6d(torch.eye(3)).repeat(24)
        with torch.no_grad():
            self.pose.bias.copy_(identity)
            self.joints.bias.zero_()

    def forward(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = token.shape[:-1]
        hidden = self.shared(token.reshape(-1, 1024))
        pose = canonical_rotation_6d(self.pose(hidden)).reshape(*shape, 24, 6)
        joints = self.joints(hidden).reshape(*shape, 17, 3)
        pelvis = joints[..., [12, 11], :].mean(dim=-2, keepdim=True)
        return pose, joints - pelvis


@dataclass(frozen=True)
class Observation:
    x: np.ndarray
    mask: np.ndarray
    crop_box: np.ndarray
    confidence: float
    valid: bool
    target_iou: float


def bbox_xyxy(box: np.ndarray) -> np.ndarray:
    center_x, center_y, side = (float(value) for value in box)
    half = side / 2.0
    return np.asarray(
        [center_x - half, center_y - half, center_x + half, center_y + half],
        dtype=np.float32,
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[:2], second[:2])
    right_bottom = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(right_bottom - left_top, 0.0)))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    return intersection / max(first_area + second_area - intersection, 1e-8)


def empty_observation() -> Observation:
    return Observation(
        np.zeros(37, dtype=np.float32),
        np.ones(17, dtype=np.bool_),
        np.zeros(3, dtype=np.float32),
        0.0,
        False,
        0.0,
    )


def parse_yolo_result(
    result: Any,
    width: int,
    height: int,
    target_box: np.ndarray | None,
    confidence_threshold: float,
    minimum_iou: float,
) -> Observation:
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return empty_observation()
    confidence = result.boxes.conf.detach().float().cpu().numpy()
    boxes = result.boxes.xyxy.detach().float().cpu().numpy()
    scale = np.asarray(
        [width / 640.0, height / 640.0, width / 640.0, height / 640.0],
        dtype=np.float32,
    )
    boxes = boxes * scale
    if target_box is None:
        best_index = int(np.argmax(confidence))
        target_iou = 1.0
    else:
        target_xyxy = bbox_xyxy(target_box)
        overlaps = np.asarray([box_iou(box, target_xyxy) for box in boxes])
        # Prefer the person matching this training track, then confidence.
        score = overlaps + 0.01 * confidence
        best_index = int(np.argmax(score))
        target_iou = float(overlaps[best_index])
    best_confidence = float(confidence[best_index])
    if best_confidence <= confidence_threshold or target_iou < minimum_iou:
        missing = empty_observation()
        return Observation(
            missing.x,
            missing.mask,
            missing.crop_box,
            best_confidence,
            False,
            target_iou,
        )

    keypoint_data = result.keypoints.data[best_index].detach().float().cpu().numpy()
    keypoints = keypoint_data[:, :2] * np.asarray(
        [width / 640.0, height / 640.0], dtype=np.float32
    )
    keypoint_confidence = keypoint_data[:, 2]
    visible = keypoint_confidence >= 0.3
    selected_box = boxes[best_index]
    if int(visible.sum()) >= 7:
        minimum = keypoints[visible].min(axis=0)
        maximum = keypoints[visible].max(axis=0)
        center_x, center_y = ((minimum + maximum) / 2.0).tolist()
        side = max(float(np.max(maximum - minimum)) * 1.2, 1.0)
    else:
        center_x = float((selected_box[0] + selected_box[2]) / 2.0)
        center_y = float((selected_box[1] + selected_box[3]) / 2.0)
        side = max(
            float(
                max(
                    selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
                )
            )
            * 1.05,
            1.0,
        )
    coordinates = (
        2.0 * (keypoints - np.asarray([center_x, center_y], dtype=np.float32)) / side
    )
    longest = float(max(width, height))
    location = np.asarray(
        [
            2.0 * center_x / longest - width / longest,
            2.0 * center_y / longest - height / longest,
            side / longest,
        ],
        dtype=np.float32,
    )
    x = np.concatenate((coordinates.reshape(34), location)).astype(np.float32)
    return Observation(
        x,
        (keypoint_confidence < 0.3).astype(np.bool_),
        np.asarray([center_x, center_y, side], dtype=np.float32),
        best_confidence,
        True,
        target_iou,
    )


def predict_observations(
    model: YOLO,
    paths: list[Path],
    target_boxes: np.ndarray | None,
    batch_size: int,
    device: torch.device,
    label: str,
    confidence_threshold: float = 0.5,
    minimum_iou: float = 0.2,
) -> dict[str, np.ndarray]:
    rows: list[Observation] = []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        originals: list[Image.Image] = []
        detector_images: list[Image.Image] = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                originals.append(rgb.copy())
                detector_images.append(
                    rgb.resize((640, 640), Image.Resampling.BILINEAR)
                )
        results = model.predict(
            detector_images,
            imgsz=640,
            conf=0.001,
            device=mobile_eval.model_device_argument(device),
            half=device.type == "cuda",
            verbose=False,
        )
        batch_targets = (
            [None] * len(results)
            if target_boxes is None
            else list(target_boxes[start : start + len(results)])
        )
        for original, result, target in zip(originals, results, batch_targets):
            rows.append(
                parse_yolo_result(
                    result,
                    *original.size,
                    target,
                    confidence_threshold,
                    minimum_iou,
                )
            )
        if (start // batch_size + 1) % 25 == 0 or len(rows) == len(paths):
            valid = sum(row.valid for row in rows)
            print(
                json.dumps(
                    {
                        "cache": label,
                        "frames": len(rows),
                        "total": len(paths),
                        "valid_rate": valid / max(len(rows), 1),
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    if len(rows) != len(paths):
        raise RuntimeError(f"YOLO cache length mismatch: {len(rows)} vs {len(paths)}")
    return {
        "x": np.stack([row.x for row in rows]),
        "mask": np.stack([row.mask for row in rows]),
        "crop_box": np.stack([row.crop_box for row in rows]),
        "confidence": np.asarray([row.confidence for row in rows], dtype=np.float32),
        "valid": np.asarray([row.valid for row in rows], dtype=np.bool_),
        "target_iou": np.asarray([row.target_iou for row in rows], dtype=np.float32),
    }


def training_row_paths(
    labels: dict[str, Any], tracks: list[phase3.TrainTrack], image_root: Path
) -> list[Path]:
    paths: list[Path | None] = [None] * len(feature_eval.to_numpy(labels["vid"]))
    for track in tracks:
        frame_ids = feature_eval.to_numpy(
            phase3.take_rows(labels["frame_id"], track.indices)
        ).astype(np.int64)
        resolved = phase3.sequence_paths(image_root, track.sequence, frame_ids)
        for row, path in zip(track.indices, resolved):
            if paths[int(row)] is not None:
                raise RuntimeError(f"Duplicate path assignment for parsed row {row}")
            paths[int(row)] = path
    missing = [index for index, path in enumerate(paths) if path is None]
    if missing:
        raise RuntimeError(f"No raw image mapping for {len(missing)} training rows")
    return [path for path in paths if path is not None]


def cache_training_observations(
    cache_path: Path,
    labels: dict[str, Any],
    tracks: list[phase3.TrainTrack],
    image_root: Path,
    pose_model: YOLO,
    yolo_batch_size: int,
    device: torch.device,
    provenance: dict[str, Any],
) -> tuple[list[Path], dict[str, np.ndarray]]:
    paths = training_row_paths(labels, tracks, image_root)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("provenance") == provenance and len(payload["valid"]) == len(
            paths
        ):
            print(f"Reusing verified YOLO training cache: {cache_path}", flush=True)
            return paths, {
                key: payload[key]
                for key in (
                    "x",
                    "mask",
                    "crop_box",
                    "confidence",
                    "valid",
                    "target_iou",
                )
            }
    target_boxes = phase3.compact_bbox(labels["bbox"])
    observations = predict_observations(
        pose_model,
        paths,
        target_boxes,
        yolo_batch_size,
        device,
        "3dpw_train",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"provenance": provenance, **observations}, cache_path)
    return paths, observations


def validation_indices(
    labels: dict[str, Any], image_root: Path, max_tracks: int
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
    if max_tracks > 0 and len(candidates) > max_tracks:
        positions = np.linspace(0, len(candidates) - 1, max_tracks, dtype=np.int64)
        candidates = [candidates[int(position)] for position in positions]
    if not candidates:
        raise RuntimeError("No 3DPW validation tracks matched raw images")
    return candidates


def cache_validation_observations(
    cache_path: Path,
    labels: dict[str, Any],
    indices: list[int],
    image_root: Path,
    pose_model: YOLO,
    yolo_batch_size: int,
    max_frames: int,
    device: torch.device,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("provenance") == provenance:
            print(f"Reusing verified YOLO validation cache: {cache_path}", flush=True)
            return payload["tracks"]
    cached: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices, start=1):
        video_id = str(labels["vid"][index])
        frame_ids = feature_eval.to_numpy(labels["frame_id"][index])
        available = min(len(frame_ids) - 1, len(labels["pose"][index]) - 1)
        if max_frames > 0:
            available = min(available, max_frames)
        paths = feature_eval.sequence_image_paths(
            image_root, video_id, frame_ids[: available + 1]
        )
        observations = predict_observations(
            pose_model,
            paths,
            None,
            yolo_batch_size,
            device,
            f"3dpw_val:{ordinal}/{len(indices)}:{video_id}",
            minimum_iou=0.0,
        )
        cached.append(
            {
                "index": index,
                "video_id": video_id,
                "paths": [str(path) for path in paths],
                "available": available,
                **observations,
            }
        )
    torch.save({"provenance": provenance, "tracks": cached}, cache_path)
    return cached


class DeploymentClipDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        labels: dict[str, Any],
        tracks: list[phase3.TrainTrack],
        paths: list[Path],
        observations: dict[str, np.ndarray],
        clip_length: int,
        stride: int,
        max_clips: int,
        minimum_valid_rate: float,
        seed: int,
    ) -> None:
        self.labels = labels
        self.tracks = tracks
        self.paths = paths
        self.observations = observations
        self.clip_length = clip_length
        clips: list[tuple[int, int]] = []
        for track_index, track in enumerate(tracks):
            maximum_start = len(track.indices) - (clip_length + 1)
            for start in range(0, maximum_start + 1, stride):
                rows = track.indices[start : start + clip_length + 1]
                valid = observations["valid"][rows]
                if bool(valid[0]) and float(valid.mean()) >= minimum_valid_rate:
                    clips.append((track_index, start))
        random.Random(seed).shuffle(clips)
        if max_clips > 0:
            clips = clips[:max_clips]
        if not clips:
            raise RuntimeError("No deployment-aware clips passed the detection filter")
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index, start = self.clips[index]
        rows = self.tracks[track_index].indices[start : start + self.clip_length + 1]
        images: list[torch.Tensor] = []
        for row in rows:
            row_index = int(row)
            if not self.observations["valid"][row_index]:
                images.append(torch.zeros(3, 256, 256, dtype=torch.float32))
                continue
            center_x, center_y, side = self.observations["crop_box"][row_index]
            with Image.open(self.paths[row_index]) as source:
                crop = mobile_eval.square_crop(
                    source.convert("RGB"), float(center_x), float(center_y), float(side)
                )
            images.append(mobile_eval.image_to_tensor(crop))

        raw_pose = (
            feature_eval.to_numpy(phase3.take_rows(self.labels["pose"], rows))
            .astype(np.float32)
            .reshape(len(rows), 24, 3)
        )
        pose = feature_eval.matrix_to_rotation_6d(
            feature_eval.axis_angle_to_matrix(torch.from_numpy(raw_pose))
        )
        joints = feature_eval.root_center_coco(
            feature_eval.to_numpy(
                phase3.take_rows(self.labels["joints3D"], rows)
            ).astype(np.float32)
        )
        teacher = feature_eval.to_numpy(
            phase3.take_rows(self.labels["features"], rows)
        ).astype(np.float32)
        betas = feature_eval.to_numpy(
            phase3.take_rows(self.labels["betas"], rows)
        ).astype(np.float32)[..., :10]
        return {
            "images": torch.stack(images),
            "x": torch.from_numpy(self.observations["x"][rows].copy()),
            "mask": torch.from_numpy(self.observations["mask"][rows].copy()),
            "valid": torch.from_numpy(self.observations["valid"][rows].copy()),
            "target_pose": pose,
            "target_joints": torch.from_numpy(joints),
            "target_betas": torch.from_numpy(betas),
            "teacher_features": torch.from_numpy(teacher),
        }


def configure_trainable(
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    stage: str,
) -> list[dict[str, Any]]:
    for model in (student, initializer, network):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for parameter in initializer.parameters():
        parameter.requires_grad_(True)

    groups: list[dict[str, Any]] = [
        {
            "params": list(initializer.parameters()),
            "lr": 2e-4 if stage == "initializer" else 5e-5,
        }
    ]
    if stage in ("joint", "last_stage"):
        for parameter in student.spatial_head.parameters():
            parameter.requires_grad_(True)
        adapters: list[nn.Parameter] = [network.mask_embedding]
        for module in (
            network.motion_encoder.embed_layer,
            network.motion_encoder.neural_init,
            network.integrator,
            network.motion_decoder.neural_init,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                adapters.append(parameter)
        groups.extend(
            [
                {
                    "params": [
                        p for p in student.spatial_head.parameters() if p.requires_grad
                    ],
                    "lr": 2e-5 if stage == "joint" else 5e-6,
                },
                {"params": adapters, "lr": 1e-5 if stage == "joint" else 2e-6},
            ]
        )
    if stage == "last_stage":
        backbone_parameters: list[nn.Parameter] = []
        for module in (student.backbone.stages[3], student.backbone.final_conv):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                backbone_parameters.append(parameter)
        groups.append({"params": backbone_parameters, "lr": 2e-7})
    return groups


def set_modes(
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    stage: str,
) -> None:
    student.eval()
    initializer.train()
    network.eval()
    if stage in ("joint", "last_stage"):
        student.spatial_head.train()
        network.motion_encoder.embed_layer.train()
        network.motion_encoder.neural_init.train()
        network.integrator.train()
        network.motion_decoder.neural_init.train()
    if stage == "last_stage":
        student.backbone.stages[3].train()
        student.backbone.final_conv.train()
    # cuDNN RNN backward needs the reserve buffer even when its weights are frozen.
    for regressor in (
        network.motion_encoder.regressor,
        network.motion_decoder.regressor,
    ):
        regressor.rnn.dropout = 0.0
        regressor.rnn.train()


def student_tokens(
    student: feature_eval.FastViTHMR2Student,
    images: torch.Tensor,
    valid: torch.Tensor,
    stage: str,
) -> torch.Tensor:
    flat = images.reshape(-1, 3, 256, 256)
    if stage == "initializer":
        with torch.no_grad():
            token = student(flat)
    elif stage == "joint":
        with torch.no_grad():
            feature_map = student.backbone.forward_features(flat[:, :, :, 32:-32])
        normalized = student.spatial_head(feature_map)
        token = normalized * student.target_std + student.target_mean
    elif stage == "last_stage":
        token = student(flat)
    else:
        raise ValueError(stage)
    token = token.reshape(*images.shape[:2], 1024)
    return token * valid.unsqueeze(-1).to(token.dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return (values * expanded).sum() / expanded.expand_as(values).sum().clamp_min(1.0)


def rotation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_matrix = feature_eval.rotation_6d_to_matrix(prediction.float())
    target_matrix = feature_eval.rotation_6d_to_matrix(target.float())
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


def pose_core(
    network: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
    init_joints: torch.Tensor,
    init_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    init_kp = torch.cat((init_joints.reshape(len(x), 1, 51), x[:, :1]), dim=-1)
    processed = network.preprocess(x[:, 1:].clone(), mask[:, 1:])
    predicted_joints, context = network.motion_encoder(processed, init_kp)
    integrated = integrate_deployment(network, context, features[:, 1:], valid[:, 1:])
    predicted_pose, predicted_shape, _, _ = network.motion_decoder(
        integrated, init_pose[:, :1]
    )
    return (
        predicted_pose.reshape(len(x), x.shape[1] - 1, 24, 6),
        predicted_joints,
        predicted_shape,
    )


def integrate_deployment(
    network: nn.Module,
    context: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Match WHAM_ImageStep's per-frame residual/bypass behavior exactly."""

    integrated = torch.cat((context, features), dim=-1)
    integrated = network.integrator.relu1(network.integrator.layer1(integrated))
    integrated = network.integrator.relu2(network.integrator.layer2(integrated))
    integrated = network.integrator.layer3(integrated)
    frame_valid = valid.to(integrated.dtype).unsqueeze(-1)
    return frame_valid * (integrated + context) + (1.0 - frame_valid) * context


@torch.inference_mode()
def deployment_core(
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
    """Vectorized equivalent of repeated exported WHAM_ImageStep calls."""

    processed = network.preprocess(x.clone(), mask)
    predicted_joints, context = network.motion_encoder(processed, init_kp)
    predicted_root, predicted_velocity = network.trajectory_decoder(
        context, init_root, cam_angvel
    )
    integrated = integrate_deployment(network, context, features, feature_valid)
    predicted_pose, predicted_shape, predicted_camera, predicted_contact = (
        network.motion_decoder(integrated, init_pose)
    )
    return {
        "pose": predicted_pose,
        "kp3d": predicted_joints,
        "root": predicted_root[:, 1:],
        "velocity": predicted_velocity,
        "shape": predicted_shape,
        "camera": predicted_camera,
        "contact": predicted_contact,
    }


LOSS_WEIGHTS = {
    "init_pose": 2.0,
    "init_root": 6.0,
    "init_joints": 20.0,
    "token": 0.20,
    "cosine": 0.20,
    "wham_pose": 4.0,
    "wham_root": 8.0,
    "wham_joints": 10.0,
    "wham_shape": 0.5,
}


def compute_losses(
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    batch: dict[str, torch.Tensor],
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token = student_tokens(student, batch["images"], batch["valid"], stage)
    initial_pose, initial_joints = initializer(token)
    valid = batch["valid"]
    init_error = rotation_loss(initial_pose, batch["target_pose"])
    init_joint_error = F.smooth_l1_loss(
        initial_joints.float(),
        batch["target_joints"].float(),
        beta=0.05,
        reduction="none",
    )
    target_z = (batch["teacher_features"] - student.target_mean) / student.target_std
    prediction_z = (token - student.target_mean) / student.target_std
    token_frame_error = (prediction_z - target_z).square().mean(dim=-1)
    cosine_error = 1.0 - F.cosine_similarity(
        token.float(), batch["teacher_features"].float(), dim=-1
    )
    predicted_pose, predicted_joints, predicted_shape = pose_core(
        network,
        batch["x"],
        batch["mask"],
        token,
        batch["valid"],
        initial_joints[:, 0],
        initial_pose,
    )
    wham_error = rotation_loss(predicted_pose, batch["target_pose"][:, 1:])
    wham_joint_error = F.smooth_l1_loss(
        predicted_joints.float(),
        batch["target_joints"][:, 1:].float(),
        beta=0.05,
        reduction="none",
    )
    wham_shape_error = F.smooth_l1_loss(
        predicted_shape.float(),
        batch["target_betas"][:, 1:].float(),
        beta=0.5,
    )
    losses = {
        "init_pose": masked_mean(init_error, valid),
        "init_root": masked_mean(init_error[..., 0], valid),
        "init_joints": masked_mean(init_joint_error, valid),
        "token": masked_mean(token_frame_error[:, 1:], valid[:, 1:]),
        "cosine": masked_mean(cosine_error[:, 1:], valid[:, 1:]),
        "wham_pose": wham_error.mean(),
        "wham_root": wham_error[..., 0].mean(),
        "wham_joints": wham_joint_error.mean(),
        "wham_shape": wham_shape_error,
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in losses.items())
    return total, losses


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_epoch(
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    stage: str,
    log_every: int,
) -> dict[str, float]:
    set_modes(student, initializer, network, stage)
    totals = {"loss": 0.0, **{name: 0.0 for name in LOSS_WEIGHTS}}
    samples = 0
    started = time.perf_counter()
    trainable = [
        parameter
        for model in (student, initializer, network)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    for step, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, components = compute_losses(
                student, initializer, network, batch, stage
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_size = len(batch["images"])
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


def observation_images(
    paths: list[str], crop_boxes: np.ndarray, valid: np.ndarray
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for path, box, is_valid in zip(paths, crop_boxes, valid):
        if not is_valid:
            images.append(torch.zeros(3, 256, 256, dtype=torch.float32))
            continue
        with Image.open(path) as source:
            crop = mobile_eval.square_crop(
                source.convert("RGB"), float(box[0]), float(box[1]), float(box[2])
            )
        images.append(mobile_eval.image_to_tensor(crop))
    return torch.stack(images)


@torch.inference_mode()
def validation_metrics(
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    labels: dict[str, Any],
    cached_tracks: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    student.eval()
    initializer.eval()
    network.eval()
    pose_errors: list[np.ndarray] = []
    init_errors: list[np.ndarray] = []
    shape_squared_errors: list[np.ndarray] = []
    per_track: list[dict[str, Any]] = []
    detection_frames = 0
    total_frames = 0
    for track in cached_tracks:
        tokens: list[torch.Tensor] = []
        for start in range(0, len(track["paths"]), batch_size):
            stop = min(start + batch_size, len(track["paths"]))
            images = observation_images(
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
        inputs = {
            "network": network,
            "x": x[:, 1:],
            "mask": mask[:, 1:],
            "features": token[:, 1:],
            "init_kp": torch.cat((initial_joints.reshape(1, 1, 51), x[:, :1]), dim=-1),
            "init_pose": initial_pose,
            "init_root": initial_pose[:, :, 0],
            "cam_angvel": torch.zeros(
                1, track["available"], 6, dtype=torch.float32, device=device
            ),
        }
        output = deployment_core(
            feature_valid=torch.from_numpy(track["valid"][1:]).unsqueeze(0).to(device),
            **inputs,
        )
        raw_target = feature_eval.to_numpy(labels["pose"][track["index"]]).astype(
            np.float32
        )[: track["available"] + 1]
        target = torch.from_numpy(raw_target).reshape(-1, 24, 3)
        errors = feature_eval.pose_error_against_axis_angle(
            output["pose"].squeeze(0).cpu(), target[1:]
        ).numpy()
        init_error = feature_eval.pose_error_against_axis_angle(
            initial_pose.squeeze(0).cpu(), target[:1]
        ).numpy()
        target_betas = feature_eval.to_numpy(labels["betas"][track["index"]]).astype(
            np.float32
        )[1 : track["available"] + 1, :10]
        shape_squared_error = (
            output["shape"].squeeze(0).cpu().numpy() - target_betas
        ) ** 2
        pose_errors.append(errors)
        init_errors.append(init_error)
        shape_squared_errors.append(shape_squared_error)
        detection_frames += int(track["valid"].sum())
        total_frames += len(track["valid"])
        per_track.append(
            {
                "sequence": track["video_id"],
                "frames": track["available"],
                "detection_rate": float(track["valid"].mean()),
                "pose_error_deg": float(errors.mean()),
                "root_error_deg": float(errors[:, 0].mean()),
                "body_error_deg": float(errors[:, 1:].mean()),
                "initializer_pose_error_deg": float(init_error.mean()),
                "initializer_root_error_deg": float(init_error[:, 0].mean()),
                "shape_rmse": float(np.sqrt(shape_squared_error.mean())),
            }
        )
    pose = np.concatenate(pose_errors)
    initial = np.concatenate(init_errors)
    shape_rmse = float(np.sqrt(np.concatenate(shape_squared_errors).mean()))
    score = float(pose.mean() + 2.0 * pose[:, 0].mean() + 2.0 * shape_rmse)
    return {
        "selection_score": score,
        "tracks": len(per_track),
        "frames": len(pose),
        "detection_rate": detection_frames / max(total_frames, 1),
        "pose_error_deg": float(pose.mean()),
        "root_error_deg": float(pose[:, 0].mean()),
        "body_error_deg": float(pose[:, 1:].mean()),
        "initializer_pose_error_deg": float(initial.mean()),
        "initializer_root_error_deg": float(initial[:, 0].mean()),
        "shape_rmse": shape_rmse,
        "per_track": per_track,
    }


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def checkpoint_payload(
    source: dict[str, Any],
    student: feature_eval.FastViTHMR2Student,
    initializer: DeploymentInitializer,
    network: nn.Module,
    epoch: int,
    stage: str,
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(source)
    payload.update(
        {
            "student_state_dict": cpu_state_dict(student),
            "deployment_schema": DEPLOYMENT_SCHEMA,
            "deployment_initializer_config": {"hidden_dim": initializer.hidden_dim},
            "deployment_initializer_state_dict": cpu_state_dict(initializer),
            "deployment_wham_state_dict": cpu_state_dict(network),
            "deployment_epoch": epoch,
            "deployment_stage": stage,
            "deployment_validation": validation,
            "deployment_metadata": metadata,
            "accepted": False,
            "acceptance_state": "trained_on_3dpw_train_selected_on_3dpw_validation_awaiting_test",
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


def self_test() -> None:
    initializer = DeploymentInitializer(hidden_dim=64).eval()
    token = torch.randn(2, 3, 1024)
    pose, joints = initializer(token)
    assert pose.shape == (2, 3, 24, 6)
    assert joints.shape == (2, 3, 17, 3)
    matrix = feature_eval.rotation_6d_to_matrix(pose)
    identity = matrix.transpose(-1, -2) @ matrix
    assert float((identity - torch.eye(3)).abs().max().detach()) < 1e-5
    assert float(joints[..., [12, 11], :].mean(dim=-2).abs().max().detach()) < 1e-5
    print(
        "Self-test passed: compact initializer shapes, rotations, and pelvis centering"
    )


def inspect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    missing_train = sorted(phase3.REQUIRED_TRAIN_KEYS - set(train_labels))
    missing_val = sorted((phase3.REQUIRED_VAL_KEYS | {"betas"}) - set(val_labels))
    if missing_train or missing_val:
        raise RuntimeError(
            f"Missing parsed keys: train={missing_train}, validation={missing_val}"
        )
    image_root = feature_eval.locate_image_root(args.three_dpw_root)
    sequence_root = phase3.locate_sequence_train_root(
        args.sequence_root or args.three_dpw_root
    )
    tracks = phase3.map_training_tracks(train_labels, sequence_root)
    paths = training_row_paths(train_labels, tracks, image_root)
    selected_val = validation_indices(val_labels, image_root, args.val_tracks)
    return {
        "image_root": str(image_root),
        "sequence_root": str(sequence_root),
        "training_rows": len(paths),
        "training_tracks": len(tracks),
        "validation_tracks": [str(val_labels["vid"][index]) for index in selected_val],
        "first_training_image": str(paths[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--train-parsed", type=Path, required=True)
    parser.add_argument("--val-parsed", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo26-weights", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-clips", type=int, default=1200)
    parser.add_argument("--minimum-valid-rate", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yolo-batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--initializer-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--last-stage-epochs", type=int, default=1)
    parser.add_argument("--val-tracks", type=int, default=8)
    parser.add_argument("--val-frames", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.train_parsed,
        args.val_parsed,
        args.source_checkpoint,
        args.wham_checkpoint,
        args.yolo26_weights,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    source_sha = feature_eval.sha256_file(args.source_checkpoint)
    yolo_sha = feature_eval.sha256_file(args.yolo26_weights)
    if source_sha != SOURCE_STUDENT_SHA256:
        raise RuntimeError(
            f"Expected phase-three source {SOURCE_STUDENT_SHA256}, found {source_sha}"
        )
    if yolo_sha != YOLO26_SHA256:
        raise RuntimeError(f"YOLO26 checksum mismatch: {yolo_sha}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    if commit != feature_eval.WHAM_COMMIT:
        raise RuntimeError(f"Unexpected WHAM commit: {commit}")
    inspection = inspect_inputs(args)
    print(json.dumps({"preflight": inspection}, indent=2), flush=True)
    if args.inspect_data:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    seed_everything(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    train_labels = joblib.load(args.train_parsed)
    val_labels = joblib.load(args.val_parsed)
    image_root = feature_eval.locate_image_root(args.three_dpw_root)
    sequence_root = phase3.locate_sequence_train_root(
        args.sequence_root or args.three_dpw_root
    )
    tracks = phase3.map_training_tracks(train_labels, sequence_root)
    val_indices = validation_indices(val_labels, image_root, args.val_tracks)

    print(
        "Loading YOLO26 and caching the actual deployment observations...", flush=True
    )
    pose_model = YOLO(str(args.yolo26_weights))
    train_cache_provenance = {
        "schema": 1,
        "split": "3dpw_train_target_matched_yolo26",
        "parsed_sha256": feature_eval.sha256_file(args.train_parsed),
        "yolo_sha256": yolo_sha,
        "confidence_threshold": 0.5,
        "minimum_iou": 0.2,
    }
    train_paths, train_observations = cache_training_observations(
        args.cache_dir / "train_yolo26_observations.pth",
        train_labels,
        tracks,
        image_root,
        pose_model,
        args.yolo_batch_size,
        device,
        train_cache_provenance,
    )
    val_cache_provenance = {
        "schema": 1,
        "split": "3dpw_validation_highest_confidence_yolo26",
        "parsed_sha256": feature_eval.sha256_file(args.val_parsed),
        "yolo_sha256": yolo_sha,
        "indices": val_indices,
        "max_frames": args.val_frames,
    }
    val_cache = cache_validation_observations(
        args.cache_dir / "val_yolo26_observations.pth",
        val_labels,
        val_indices,
        image_root,
        pose_model,
        args.yolo_batch_size,
        args.val_frames,
        device,
        val_cache_provenance,
    )
    del pose_model
    torch.cuda.empty_cache()

    dataset = DeploymentClipDataset(
        train_labels,
        tracks,
        train_paths,
        train_observations,
        args.clip_length,
        args.stride,
        args.max_clips,
        args.minimum_valid_rate,
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
    loader = DataLoader(dataset, **loader_options)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    student, source_training = feature_eval.load_student(args.source_checkpoint, device)
    initializer = DeploymentInitializer().to(device)
    network = feature_eval.load_wham_core(args.wham_repo, args.wham_checkpoint, device)
    metadata = {
        "schema_version": 1,
        "deployment_schema": DEPLOYMENT_SCHEMA,
        "source_checkpoint_sha256": source_sha,
        "source_training": source_training,
        "wham_commit": commit,
        "wham_checkpoint_sha256": feature_eval.sha256_file(args.wham_checkpoint),
        "yolo26_sha256": yolo_sha,
        "train_parsed_sha256": feature_eval.sha256_file(args.train_parsed),
        "val_parsed_sha256": feature_eval.sha256_file(args.val_parsed),
        "test_data_used": False,
        "training_tracks": len(tracks),
        "training_clips": len(dataset),
        "training_detection_rate": float(train_observations["valid"].mean()),
        "clip_length": args.clip_length,
        "stride": args.stride,
        "loss_weights": LOSS_WEIGHTS,
        "trainable_wham_parts": [
            "mask_embedding",
            "motion_encoder.embed_layer",
            "motion_encoder.neural_init",
            "integrator",
            "motion_decoder.neural_init",
        ],
        "frozen_wham_recurrent_weights": True,
        "seed": args.seed,
    }
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_validation: dict[str, Any] | None = None
    best_epoch = 0
    best_stage = "none"
    best_path = args.output_dir / "tiny_pipeline_best.pth"
    stages = (
        ("initializer", args.initializer_epochs),
        ("joint", args.joint_epochs),
        ("last_stage", args.last_stage_epochs),
    )
    epoch = 0
    for stage, epochs in stages:
        if epochs <= 0:
            continue
        groups = configure_trainable(student, initializer, network, stage)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        total_steps = max(len(loader) * epochs, 1)
        warmup = min(len(loader), max(total_steps // 10, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_steps, warm=warmup: cosine_schedule(
                step, total, warm
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for _ in range(epochs):
            epoch += 1
            epoch_started = time.perf_counter()
            training = train_epoch(
                student,
                initializer,
                network,
                loader,
                optimizer,
                scheduler,
                scaler,
                device,
                stage,
                args.log_every,
            )
            validation = validation_metrics(
                student,
                initializer,
                network,
                val_labels,
                val_cache,
                args.feature_batch_size,
                device,
            )
            row = {
                "epoch": epoch,
                "stage": stage,
                **{f"train_{key}": value for key, value in training.items()},
                **{
                    f"val_{key}": value
                    for key, value in validation.items()
                    if key != "per_track"
                },
                "minutes": (time.perf_counter() - epoch_started) / 60.0,
            }
            history.append(row)
            print(json.dumps(row, indent=2), flush=True)
            if validation["selection_score"] < best_score:
                best_score = validation["selection_score"]
                best_validation = validation
                best_epoch = epoch
                best_stage = stage
                torch.save(
                    checkpoint_payload(
                        source,
                        student,
                        initializer,
                        network,
                        epoch,
                        stage,
                        validation,
                        metadata,
                    ),
                    best_path,
                )
                print(
                    f"Selected new best deployment checkpoint: {best_path}", flush=True
                )
            write_history(args.output_dir / "deployment_training_history.csv", history)

    if best_validation is None or not best_path.is_file():
        raise RuntimeError("Training produced no validation checkpoint")
    report = {
        "schema_version": 1,
        "deployment_schema": DEPLOYMENT_SCHEMA,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_selection_score": best_score,
        "best_validation": best_validation,
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": feature_eval.sha256_file(best_path),
        "metadata": metadata,
        "next_action": "run_the_locked_3dpw_test_once_and_do_not_tune_from_it",
    }
    report_path = args.output_dir / "deployment_training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_stage": best_stage,
                "best_validation": {
                    key: value
                    for key, value in best_validation.items()
                    if key != "per_track"
                },
                "checkpoint_sha256": report["best_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
