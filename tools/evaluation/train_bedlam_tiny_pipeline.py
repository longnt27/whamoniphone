#!/usr/bin/env python3
"""BEDLAM adaptation and validation-locked calibration for the tiny WHAM path.

The program starts from the previously evaluated deployment checkpoint, streams
licensed BEDLAM MP4 scene archives from the official Hugging Face mirror, and
uses the uploaded SMPL labels without materializing the PNG release.  It then
calibrates on 3DPW train and selects only on 3DPW validation.  The 3DPW test set
is intentionally not accepted by this training program.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import evaluate_deployment_tiny_pipeline_3dpw as deployment_eval
import evaluate_mobile_pipeline_3dpw as mobile_eval
import evaluate_wham_feature_substitution as feature_eval
import finetune_fastvit_wham_downstream as phase3
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from ultralytics import YOLO

import train_deployment_tiny_pipeline as deployment


SOURCE_DEPLOYMENT_SHA256 = (
    "d47ac0c855f44f84b67b20b9abb2cd3a66af702de479692350723cd79815c2e4"
)
HF_REPO_DEFAULT = "Intelligent-Systems/BEDLAM"
OUTPUT_CHECKPOINT = "bedlam_tiny_pipeline_best.pth"
BEDLAM_LOSS_WEIGHTS = {
    "init_pose": 2.0,
    "init_root": 5.0,
    "wham_pose": 3.0,
    "wham_root": 6.0,
    "wham_shape": 0.35,
    "token_cosine": 0.08,
    "joint_preserve": 0.50,
}


@dataclass(frozen=True)
class RemoteScene:
    label_path: Path
    label_scene: str
    remote_path: str
    remote_size: int


@dataclass
class BedlamSample:
    crop: np.ndarray
    x: np.ndarray
    mask: np.ndarray
    pose: np.ndarray
    shape: np.ndarray
    frame_id: int
    track_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_scene_name(path: Path) -> str:
    return re.sub(r"_(?:6|30)fps$", "", path.stem)


def expected_remote_candidates(path: Path) -> list[str]:
    scene = label_scene_name(path)
    archive = scene + "_mp4.tar"
    # The official Intelligent-Systems mirror keeps each release component
    # below a scene directory.  Older third-party mirrors used the flat
    # videos/ layout, so retain those paths only as compatibility fallbacks.
    return [
        f"{scene}/mp4/{archive}",
        f"videos/{archive}",
        f"movies/{archive}",
        archive,
    ]


def frame_id_from_name(name: str) -> int:
    match = re.search(r"_(\d+)\.[^.]+$", PurePosixPath(name).name)
    if match is None:
        raise ValueError(f"Could not parse a BEDLAM frame number from {name!r}")
    return int(match.group(1))


def evenly_spaced(values: list[str], maximum: int) -> list[str]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return [values[int(position)] for position in positions]


def centered_window(values: list[int], maximum: int) -> list[int]:
    """Keep a contiguous portion so six-fps temporal clips remain contiguous."""

    if maximum <= 0 or len(values) <= maximum:
        return values
    start = (len(values) - maximum) // 2
    return values[start : start + maximum]


def scene_category(name: str) -> str:
    for marker in (
        "closeup",
        "hair",
        "highbmi",
        "orbit",
        "stadium",
        "highSchoolGym",
        "bigOffice",
        "suburb",
    ):
        if marker.lower() in name.lower():
            return marker.lower()
    return "general"


def resolve_remote_scenes(
    label_paths: list[Path], repo_id: str, token: str
) -> tuple[list[RemoteScene], list[dict[str, Any]]]:
    api = HfApi(token=token)
    try:
        root_entries = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision="main",
                recursive=False,
                expand=False,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not list the gated BEDLAM repository root: {error}"
        ) from error
    root_paths = {str(entry.path).strip("/") for entry in root_entries}
    print(
        json.dumps(
            {
                "preflight": "huggingface_repository_layout",
                "root_entries": len(root_paths),
                "matching_scene_directories": sum(
                    label_scene_name(path) in root_paths for path in label_paths
                ),
            }
        ),
        flush=True,
    )
    resolved: list[RemoteScene] = []
    missing: list[dict[str, Any]] = []
    for ordinal, label_path in enumerate(label_paths, start=1):
        candidates = expected_remote_candidates(label_path)
        remote_path = None
        remote_size = 0
        scene = label_scene_name(label_path)

        # One directory listing both verifies the official path and returns
        # its size.  This replaces several slow negative HEAD requests per
        # scene and makes a layout error visible immediately.
        if scene in root_paths:
            try:
                mp4_entries = list(
                    api.list_repo_tree(
                        repo_id=repo_id,
                        path_in_repo=f"{scene}/mp4",
                        repo_type="dataset",
                        revision="main",
                        recursive=False,
                        expand=True,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    f"Could not list gated BEDLAM directory {scene}/mp4: {error}"
                ) from error
            expected = candidates[0]
            match = next(
                (
                    entry
                    for entry in mp4_entries
                    if str(getattr(entry, "path", "")) == expected
                ),
                None,
            )
            if match is not None:
                remote_path = expected
                remote_size = int(getattr(match, "size", 0) or 0)

        # Compatibility only for a caller that intentionally overrides the
        # official repository with an older mirror.
        if remote_path is None and repo_id != HF_REPO_DEFAULT:
            for candidate in candidates[1:]:
                try:
                    exists = api.file_exists(
                        repo_id=repo_id,
                        filename=candidate,
                        repo_type="dataset",
                        revision="main",
                    )
                except Exception as error:  # gated access and transient API errors
                    raise RuntimeError(
                        f"Could not query gated BEDLAM file {candidate}: {error}"
                    ) from error
                if exists:
                    remote_path = candidate
                    break
        if remote_path is None:
            missing.append({"label": label_path.name, "tried": candidates})
            continue
        if remote_size <= 0:
            metadata = get_hf_file_metadata(
                hf_hub_url(repo_id, remote_path, repo_type="dataset"), token=token
            )
            remote_size = int(metadata.size or 0)
        resolved.append(
            RemoteScene(
                label_path=label_path,
                label_scene=scene,
                remote_path=remote_path,
                remote_size=remote_size,
            )
        )
        if ordinal % 8 == 0 or ordinal == len(label_paths):
            print(
                json.dumps(
                    {
                        "preflight": "huggingface_scene_resolution",
                        "checked": ordinal,
                        "labels": len(label_paths),
                        "matched": len(resolved),
                    }
                ),
                flush=True,
            )
    return resolved, missing


def select_scenes(
    scenes: list[RemoteScene], maximum_scenes: int, maximum_download_gib: float
) -> list[RemoteScene]:
    if not scenes:
        raise RuntimeError("No BEDLAM label file matched an official MP4 archive")
    by_category: dict[str, list[RemoteScene]] = defaultdict(list)
    for scene in sorted(scenes, key=lambda item: (item.remote_size, item.label_scene)):
        by_category[scene_category(scene.label_scene)].append(scene)
    ordered: list[RemoteScene] = []
    while any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category]:
                ordered.append(by_category[category].pop(0))
    limit = int(maximum_download_gib * (1024**3))
    selected: list[RemoteScene] = []
    total = 0
    for scene in ordered:
        if maximum_scenes > 0 and len(selected) >= maximum_scenes:
            break
        if selected and limit > 0 and total + scene.remote_size > limit:
            continue
        selected.append(scene)
        total += scene.remote_size
    if not selected:
        # Always permit the smallest archive so a conservative limit cannot
        # accidentally turn a paid GPU run into a no-op.
        selected = [min(scenes, key=lambda item: item.remote_size)]
    return selected


def validate_label_file(path: Path) -> dict[str, Any]:
    required = {
        "imgname",
        "center",
        "scale",
        "pose_cam",
        "shape",
        "gtkps",
        "cam_int",
        "cam_ext",
        "gender",
    }
    with zipfile.ZipFile(path) as archive:
        available = {
            PurePosixPath(name).stem
            for name in archive.namelist()
            if name.endswith(".npy")
        }
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"{path.name} is missing BEDLAM keys {missing}")
        shapes: dict[str, tuple[int, ...]] = {}
        for key in ("pose_cam", "shape"):
            with archive.open(f"{key}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise RuntimeError(
                        f"Unsupported NPY version {version} in {path.name}:{key}"
                    )
                shapes[key] = tuple(int(value) for value in shape)
    with np.load(path, allow_pickle=False) as labels:
        rows = len(labels["imgname"])
        if shapes["pose_cam"] != (rows, 72):
            raise RuntimeError(
                f"{path.name} pose_cam has shape {shapes['pose_cam']}, "
                f"expected {(rows, 72)}"
            )
        if shapes["shape"][0] != rows or shapes["shape"][1] < 10:
            raise RuntimeError(f"Unexpected shape array in {path.name}")
        names = labels["imgname"]
        sequence_count = len({str(PurePosixPath(str(name)).parent) for name in names})
        first_ids = [frame_id_from_name(str(name)) for name in names[: min(rows, 5000)]]
        if any(frame % 5 for frame in first_ids):
            raise RuntimeError(f"{path.name} is not the expected every-fifth-frame label set")
    return {"file": path.name, "rows": rows, "sequences": sequence_count}


def map_sequence_rows(names: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, raw_name in enumerate(names):
        grouped[PurePosixPath(str(raw_name)).parent.name].append(row)
    return {
        sequence: np.asarray(rows, dtype=np.int64)
        for sequence, rows in grouped.items()
    }


def tar_video_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".mp4"):
            continue
        stem = PurePosixPath(member.name).stem
        if stem in members:
            raise RuntimeError(f"Duplicate MP4 stem {stem!r} in {archive.name}")
        members[stem] = member
    if not members:
        raise RuntimeError(f"No MP4 files were found in {archive.name}")
    return members


def extract_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path
) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Could not read {member.name} from {archive.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def decode_selected_frames(
    video_path: Path, frame_ids: list[int], rotate_clockwise: bool
) -> dict[int, Image.Image]:
    wanted = set(frame_ids)
    if not wanted:
        return {}
    maximum = max(wanted)
    decoded: dict[int, Image.Image] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    index = 0
    while index <= maximum:
        success, frame = capture.read()
        if not success:
            break
        if index in wanted:
            if rotate_clockwise:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = Image.fromarray(rgb)
        index += 1
    capture.release()
    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(
            f"{video_path.name} ended before requested frames {missing[:10]}"
        )
    return decoded


def shape_track_id(shape: np.ndarray, gender: str) -> str:
    canonical = np.asarray(shape, dtype="<f4")
    return gender.lower() + ":" + hashlib.sha1(canonical.tobytes()).hexdigest()


def samples_from_video(
    labels: Any,
    rows: np.ndarray,
    video_path: Path,
    pose_model: YOLO,
    device: torch.device,
    maximum_frames: int,
    yolo_batch_size: int,
    rotate_clockwise: bool,
    confidence_threshold: float = 0.25,
    minimum_iou: float = 0.20,
) -> tuple[list[BedlamSample], dict[str, Any]]:
    raw_names = labels["imgname"][rows]
    frame_to_rows: dict[int, list[int]] = defaultdict(list)
    for row, name in zip(rows.tolist(), raw_names.tolist()):
        frame_to_rows[frame_id_from_name(str(name))].append(int(row))
    chosen_frames = centered_window(sorted(frame_to_rows), maximum_frames)
    decoded = decode_selected_frames(video_path, chosen_frames, rotate_clockwise)
    samples: list[BedlamSample] = []
    attempted = sum(len(frame_to_rows[frame]) for frame in chosen_frames)
    for start in range(0, len(chosen_frames), yolo_batch_size):
        batch_ids = chosen_frames[start : start + yolo_batch_size]
        originals = [decoded[frame] for frame in batch_ids]
        detector_images = [
            image.resize((640, 640), Image.Resampling.BILINEAR)
            for image in originals
        ]
        results = pose_model.predict(
            detector_images,
            imgsz=640,
            conf=0.001,
            device=mobile_eval.model_device_argument(device),
            half=device.type == "cuda",
            verbose=False,
        )
        for frame, original, result in zip(batch_ids, originals, results):
            width, height = original.size
            for row in frame_to_rows[frame]:
                target_box = np.asarray(
                    [
                        float(labels["center"][row, 0]),
                        float(labels["center"][row, 1]),
                        float(labels["scale"][row]) * 200.0,
                    ],
                    dtype=np.float32,
                )
                observation = deployment.parse_yolo_result(
                    result,
                    width,
                    height,
                    target_box,
                    confidence_threshold=confidence_threshold,
                    minimum_iou=minimum_iou,
                )
                if not observation.valid:
                    continue
                center_x, center_y, side = observation.crop_box
                crop = mobile_eval.square_crop(
                    original,
                    float(center_x),
                    float(center_y),
                    float(side),
                )
                crop_array = np.asarray(crop, dtype=np.uint8).copy()
                gender = str(labels["gender"][row])
                samples.append(
                    BedlamSample(
                        crop=crop_array,
                        x=observation.x.copy(),
                        mask=observation.mask.copy(),
                        pose=np.asarray(labels["pose_cam"][row], dtype=np.float32)
                        .reshape(24, 3)
                        .copy(),
                        # WHAM predicts the conventional ten SMPL betas.  The
                        # uploaded labels contain at least ten and sometimes 11.
                        shape=np.asarray(labels["shape"][row, :10], dtype=np.float32)
                        .copy(),
                        frame_id=frame,
                        track_id=shape_track_id(labels["shape"][row], gender),
                    )
                )
        print(
            json.dumps(
                {
                    "bedlam_video": video_path.name,
                    "decoded_frames": min(start + len(batch_ids), len(chosen_frames)),
                    "requested_frames": len(chosen_frames),
                    "valid_person_samples": len(samples),
                }
            ),
            flush=True,
        )
    return samples, {
        "attempted_person_frames": attempted,
        "accepted_person_frames": len(samples),
        "detection_rate": len(samples) / max(attempted, 1),
        "unique_frames": len(chosen_frames),
    }


def build_temporal_clips(
    samples: list[BedlamSample], clip_length: int, stride: int, maximum: int
) -> list[list[BedlamSample]]:
    grouped: dict[str, list[BedlamSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.track_id].append(sample)
    clips: list[list[BedlamSample]] = []
    for track_samples in grouped.values():
        ordered = sorted(track_samples, key=lambda sample: sample.frame_id)
        # Exact shape fingerprints should identify one BEDLAM actor.  A duplicate
        # frame would indicate an ambiguous identity and is therefore excluded.
        by_frame: dict[int, BedlamSample] = {}
        ambiguous: set[int] = set()
        for sample in ordered:
            if sample.frame_id in by_frame:
                ambiguous.add(sample.frame_id)
            else:
                by_frame[sample.frame_id] = sample
        ordered = [
            sample
            for frame, sample in sorted(by_frame.items())
            if frame not in ambiguous
        ]
        runs: list[list[BedlamSample]] = []
        current: list[BedlamSample] = []
        for sample in ordered:
            if current and sample.frame_id - current[-1].frame_id != 5:
                runs.append(current)
                current = []
            current.append(sample)
        if current:
            runs.append(current)
        for run in runs:
            for start in range(0, len(run) - clip_length + 1, stride):
                clips.append(run[start : start + clip_length])
    random.shuffle(clips)
    if maximum > 0:
        clips = clips[:maximum]
    return clips


def normalized_images(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array).to(device, non_blocking=True).float() / 255.0
    if tensor.ndim == 5:
        tensor = tensor.permute(0, 1, 4, 2, 3)
        mean = mobile_eval.IMAGENET_MEAN.to(device).reshape(1, 1, 3, 1, 1)
        std = mobile_eval.IMAGENET_STD.to(device).reshape(1, 1, 3, 1, 1)
    elif tensor.ndim == 4:
        tensor = tensor.permute(0, 3, 1, 2)
        mean = mobile_eval.IMAGENET_MEAN.to(device).reshape(1, 3, 1, 1)
        std = mobile_eval.IMAGENET_STD.to(device).reshape(1, 3, 1, 1)
    else:
        raise ValueError(f"Unexpected crop batch shape {tuple(tensor.shape)}")
    return (tensor - mean) / std


def configure_bedlam_trainable(
    student: feature_eval.FastViTHMR2Student,
    initializer: deployment.DeploymentInitializer,
    network: nn.Module,
) -> torch.optim.Optimizer:
    for model in (student, initializer, network):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for parameter in initializer.parameters():
        parameter.requires_grad_(True)
    for parameter in student.spatial_head.parameters():
        parameter.requires_grad_(True)
    backbone_parameters: list[nn.Parameter] = []
    for module in (student.backbone.stages[3], student.backbone.final_conv):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            backbone_parameters.append(parameter)
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
    groups = [
        {"params": list(initializer.parameters()), "lr": 3e-5},
        {"params": list(student.spatial_head.parameters()), "lr": 1e-5},
        {"params": backbone_parameters, "lr": 1e-7},
        {"params": adapters, "lr": 4e-6},
    ]
    return torch.optim.AdamW(groups, weight_decay=0.01)


def set_bedlam_modes(
    student: feature_eval.FastViTHMR2Student,
    initializer: deployment.DeploymentInitializer,
    network: nn.Module,
) -> None:
    deployment.set_modes(student, initializer, network, "last_stage")


def bedlam_frame_loss(
    student: feature_eval.FastViTHMR2Student,
    initializer: deployment.DeploymentInitializer,
    teacher_student: feature_eval.FastViTHMR2Student,
    teacher_initializer: deployment.DeploymentInitializer,
    samples: list[BedlamSample],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = normalized_images(
        np.stack([sample.crop for sample in samples]), device
    )
    target_axis = torch.from_numpy(
        np.stack([sample.pose for sample in samples])
    ).to(device)
    target_pose = feature_eval.matrix_to_rotation_6d(
        feature_eval.axis_angle_to_matrix(target_axis)
    )
    token = student(images)
    predicted_pose, predicted_joints = initializer(token)
    with torch.no_grad():
        teacher_token = teacher_student(images)
        _, teacher_joints = teacher_initializer(teacher_token)
    pose_error = deployment.rotation_loss(predicted_pose, target_pose)
    components = {
        "init_pose": pose_error.mean(),
        "init_root": pose_error[:, 0].mean(),
        "token_cosine": (
            1.0 - F.cosine_similarity(token.float(), teacher_token.float(), dim=-1)
        ).mean(),
        "joint_preserve": F.smooth_l1_loss(
            predicted_joints.float(), teacher_joints.float(), beta=0.05
        ),
    }
    total = sum(BEDLAM_LOSS_WEIGHTS[name] * value for name, value in components.items())
    return total, components


def bedlam_clip_loss(
    student: feature_eval.FastViTHMR2Student,
    initializer: deployment.DeploymentInitializer,
    network: nn.Module,
    teacher_student: feature_eval.FastViTHMR2Student,
    teacher_initializer: deployment.DeploymentInitializer,
    clips: list[list[BedlamSample]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    crop_array = np.stack(
        [[sample.crop for sample in clip] for clip in clips]
    )
    images = normalized_images(crop_array, device)
    batch_size, sequence_length = images.shape[:2]
    flat = images.reshape(-1, 3, 256, 256)
    token = student(flat).reshape(batch_size, sequence_length, 1024)
    predicted_initial_pose, predicted_initial_joints = initializer(token)
    with torch.no_grad():
        teacher_token = teacher_student(flat).reshape(batch_size, sequence_length, 1024)
        _, teacher_joints = teacher_initializer(teacher_token)
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
    valid = torch.ones(
        batch_size, sequence_length, dtype=torch.bool, device=device
    )
    predicted_pose, _, predicted_shape = deployment.pose_core(
        network,
        x,
        mask,
        token,
        valid,
        predicted_initial_joints[:, 0],
        predicted_initial_pose,
    )
    init_error = deployment.rotation_loss(predicted_initial_pose, target_pose)
    wham_error = deployment.rotation_loss(predicted_pose, target_pose[:, 1:])
    components = {
        "init_pose": init_error.mean(),
        "init_root": init_error[..., 0].mean(),
        "wham_pose": wham_error.mean(),
        "wham_root": wham_error[..., 0].mean(),
        "wham_shape": F.smooth_l1_loss(
            predicted_shape.float(), target_shape[:, 1:].float(), beta=0.5
        ),
        "token_cosine": (
            1.0 - F.cosine_similarity(token.float(), teacher_token.float(), dim=-1)
        ).mean(),
        "joint_preserve": F.smooth_l1_loss(
            predicted_initial_joints.float(), teacher_joints.float(), beta=0.05
        ),
    }
    total = sum(BEDLAM_LOSS_WEIGHTS[name] * value for name, value in components.items())
    return total, components


def optimize_batches(
    batches: Iterable[list[Any]],
    loss_function: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    trainable: list[nn.Parameter],
    log_label: str,
    log_every: int,
) -> tuple[dict[str, float], int]:
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    started = time.perf_counter()
    for step, batch in enumerate(batches, start=1):
        if not batch:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            loss, components = loss_function(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        count += 1
        totals["loss"] += float(loss.detach())
        for name, value in components.items():
            totals[name] += float(value.detach())
        if step % log_every == 0:
            print(
                json.dumps(
                    {
                        "training": log_label,
                        "steps": step,
                        "mean_loss": totals["loss"] / count,
                        "minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    return ({key: value / max(count, 1) for key, value in totals.items()}, count)


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def compact_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in validation.items() if key != "per_track"}


def save_candidate(
    source: dict[str, Any],
    student: feature_eval.FastViTHMR2Student,
    initializer: deployment.DeploymentInitializer,
    network: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    stage: str,
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    torch.save(
        deployment.checkpoint_payload(
            source,
            student,
            initializer,
            network,
            epoch,
            stage,
            validation,
            metadata,
        ),
        checkpoint_path,
    )


def load_selected_models(
    checkpoint_path: Path,
    wham_repo: Path,
    wham_checkpoint: Path,
    device: torch.device,
) -> tuple[
    feature_eval.FastViTHMR2Student,
    deployment.DeploymentInitializer,
    nn.Module,
]:
    student, initializer, network, _ = deployment_eval.load_deployment_models(
        checkpoint_path, wham_repo, wham_checkpoint, device
    )
    return student, initializer, network


def locate_tar_member(
    members: dict[str, tarfile.TarInfo], sequence: str
) -> tarfile.TarInfo | None:
    candidates = [sequence, sequence.replace("seq_", "")]
    for candidate in candidates:
        if candidate in members:
            return members[candidate]
    suffix_matches = [
        member for stem, member in members.items() if stem.endswith(sequence)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def self_test() -> None:
    assert label_scene_name(Path("example_6fps.npz")) == "example"
    assert label_scene_name(Path("example_30fps.npz")) == "example"
    assert expected_remote_candidates(Path("example_6fps.npz"))[0] == (
        "example/mp4/example_mp4.tar"
    )
    assert frame_id_from_name("seq_000001/seq_000001_0125.png") == 125
    samples = [
        BedlamSample(
            crop=np.zeros((256, 256, 3), dtype=np.uint8),
            x=np.zeros(37, dtype=np.float32),
            mask=np.zeros(17, dtype=np.bool_),
            pose=np.zeros((24, 3), dtype=np.float32),
            shape=np.zeros(10, dtype=np.float32),
            frame_id=frame,
            track_id="actor",
        )
        for frame in (0, 5, 10, 15, 30, 35)
    ]
    clips = build_temporal_clips(samples, clip_length=3, stride=1, maximum=0)
    assert sorted([sample.frame_id for sample in clip] for clip in clips) == [
        [0, 5, 10],
        [5, 10, 15],
    ]
    print("Self-test passed: BEDLAM scene mapping, frame parsing, and track clips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bedlam-label-root", type=Path, required=False)
    parser.add_argument("--hf-repo", default=HF_REPO_DEFAULT)
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
    parser.add_argument("--maximum-scenes", type=int, default=6)
    parser.add_argument("--maximum-download-gib", type=float, default=2.0)
    parser.add_argument("--videos-per-scene", type=int, default=32)
    parser.add_argument("--frames-per-video", type=int, default=30)
    parser.add_argument("--clips-per-video", type=int, default=12)
    parser.add_argument("--bedlam-clip-length", type=int, default=8)
    parser.add_argument("--bedlam-clip-stride", type=int, default=4)
    parser.add_argument("--bedlam-batch-size", type=int, default=2)
    parser.add_argument("--bedlam-frame-batch-size", type=int, default=24)
    parser.add_argument("--yolo-batch-size", type=int, default=32)
    parser.add_argument("--three-dpw-clip-length", type=int, default=24)
    parser.add_argument("--three-dpw-stride", type=int, default=12)
    parser.add_argument("--three-dpw-max-clips", type=int, default=1200)
    parser.add_argument("--three-dpw-batch-size", type=int, default=2)
    parser.add_argument("--three-dpw-joint-epochs", type=int, default=2)
    parser.add_argument("--three-dpw-last-stage-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--val-tracks", type=int, default=8)
    parser.add_argument("--val-frames", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def require_paths(args: argparse.Namespace) -> None:
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
    )
    if any(path is None for path in required):
        raise ValueError("All data/model path arguments are required outside --self-test")
    files = (
        args.train_parsed,
        args.val_parsed,
        args.source_checkpoint,
        args.wham_checkpoint,
        args.yolo26_weights,
        args.wham_repo / "lib/models/wham.py",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    require_paths(args)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it as a Kaggle secret and expose it to this run."
        )
    source_sha = sha256_file(args.source_checkpoint)
    if source_sha != SOURCE_DEPLOYMENT_SHA256:
        raise RuntimeError(
            f"Expected deployment checkpoint {SOURCE_DEPLOYMENT_SHA256}, found {source_sha}"
        )
    if sha256_file(args.yolo26_weights) != deployment.YOLO26_SHA256:
        raise RuntimeError("YOLO26 checkpoint checksum mismatch")
    label_paths = sorted(args.bedlam_label_root.glob("*.npz"))
    label_paths = [path for path in label_paths if path.name != "agora.npz"]
    if len(label_paths) < 20:
        raise RuntimeError(
            f"Expected the BEDLAM SMPL label package below {args.bedlam_label_root}; "
            f"found only {len(label_paths)} scene files"
        )
    label_manifest = []
    for ordinal, path in enumerate(label_paths, start=1):
        label_manifest.append(validate_label_file(path))
        print(
            json.dumps(
                {
                    "preflight": "local_bedlam_labels",
                    "checked": ordinal,
                    "files": len(label_paths),
                    "rows_seen": sum(item["rows"] for item in label_manifest),
                }
            ),
            flush=True,
        )
    scenes, unmatched = resolve_remote_scenes(label_paths, args.hf_repo, token)
    selected_scenes = select_scenes(
        scenes, args.maximum_scenes, args.maximum_download_gib
    )
    preflight = {
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": source_sha,
        "bedlam_label_root": str(args.bedlam_label_root),
        "bedlam_label_files": len(label_paths),
        "bedlam_label_rows": sum(row["rows"] for row in label_manifest),
        "hf_repo": args.hf_repo,
        "matched_scene_archives": len(scenes),
        "unmatched_labels": unmatched,
        "selected_scenes": [
            {
                "label": scene.label_path.name,
                "remote": scene.remote_path,
                "download_gib": scene.remote_size / (1024**3),
            }
            for scene in selected_scenes
        ],
        "selected_download_gib": sum(scene.remote_size for scene in selected_scenes)
        / (1024**3),
        "shape_adapter": "first_10_of_at_least_10_bedlam_values",
        "effective_bedlam_label_rate_hz": 6,
        "closeup_rotation": "90_degrees_clockwise",
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
    val_indices = deployment.validation_indices(
        val_labels, image_root, args.val_tracks
    )

    print("Loading YOLO26 and caching 3DPW validation observations...", flush=True)
    pose_model = YOLO(str(args.yolo26_weights))
    val_cache = deployment.cache_validation_observations(
        args.cache_dir / "val_yolo26_observations.pth",
        val_labels,
        val_indices,
        image_root,
        pose_model,
        args.yolo_batch_size,
        args.val_frames,
        device,
        {
            "schema": 2,
            "split": "3dpw_validation_highest_confidence_yolo26",
            "parsed_sha256": sha256_file(args.val_parsed),
            "yolo_sha256": deployment.YOLO26_SHA256,
            "indices": val_indices,
            "max_frames": args.val_frames,
        },
    )

    student, initializer, network = load_selected_models(
        args.source_checkpoint, args.wham_repo, args.wham_checkpoint, device
    )
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    metadata = dict(source.get("deployment_metadata") or {})
    metadata.update(
        {
            "bedlam_adaptation_schema": 1,
            "source_deployment_checkpoint_sha256": source_sha,
            "bedlam_hf_repo": args.hf_repo,
            "bedlam_selected_scene_archives": [
                scene.remote_path for scene in selected_scenes
            ],
            "bedlam_loss_weights": BEDLAM_LOSS_WEIGHTS,
            "bedlam_effective_label_rate_hz": 6,
            "bedlam_closeup_rotation_clockwise_degrees": 90,
            "bedlam_shape_values_used": 10,
            "bedlam_detector_conditioned_crops": True,
            "frozen_wham_recurrent_weights": True,
            "test_data_used": False,
            "prior_3dpw_test_result_known": True,
            "selection_split": "3dpw_validation_only",
        }
    )
    best_path = args.output_dir / OUTPUT_CHECKPOINT
    baseline_validation = deployment.validation_metrics(
        student,
        initializer,
        network,
        val_labels,
        val_cache,
        args.feature_batch_size,
        device,
    )
    best_score = float(baseline_validation["selection_score"])
    best_validation = baseline_validation
    best_stage = "source_deployment_baseline"
    best_epoch = 0
    save_candidate(
        source,
        student,
        initializer,
        network,
        best_path,
        best_epoch,
        best_stage,
        best_validation,
        metadata,
    )
    history: list[dict[str, Any]] = [
        {
            "phase": "baseline",
            "epoch": 0,
            "scene": "",
            **{
                f"val_{key}": value
                for key, value in compact_validation(baseline_validation).items()
            },
            "selected": True,
        }
    ]
    print(
        json.dumps({"baseline_validation": compact_validation(baseline_validation)}, indent=2),
        flush=True,
    )

    teacher_student = copy.deepcopy(student).eval()
    teacher_initializer = copy.deepcopy(initializer).eval()
    for model in (teacher_student, teacher_initializer):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    optimizer = configure_bedlam_trainable(student, initializer, network)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    trainable = [
        parameter
        for model in (student, initializer, network)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    bedlam_manifest: list[dict[str, Any]] = []
    global_epoch = 0
    for scene_ordinal, scene in enumerate(selected_scenes, start=1):
        global_epoch += 1
        scene_started = time.perf_counter()
        scene_cache = args.scratch_dir / f"hf_scene_{scene_ordinal:02d}"
        if scene_cache.exists():
            shutil.rmtree(scene_cache)
        print(
            f"Downloading BEDLAM scene {scene_ordinal}/{len(selected_scenes)}: "
            f"{scene.remote_path}",
            flush=True,
        )
        tar_path = Path(
            hf_hub_download(
                repo_id=args.hf_repo,
                filename=scene.remote_path,
                repo_type="dataset",
                revision="main",
                token=token,
                cache_dir=scene_cache,
            )
        )
        scene_stats: defaultdict[str, float] = defaultdict(float)
        scene_losses: defaultdict[str, float] = defaultdict(float)
        scene_steps = 0
        with np.load(scene.label_path, allow_pickle=False) as labels, tarfile.open(
            tar_path, mode="r:*"
        ) as archive:
            # NpzFile does not cache arrays: materialize each required field once
            # so per-person indexing never re-reads a multi-hundred-MB member.
            scene_labels = {
                key: labels[key]
                for key in (
                    "imgname",
                    "center",
                    "scale",
                    "pose_cam",
                    "shape",
                    "gender",
                )
            }
            sequence_rows = map_sequence_rows(scene_labels["imgname"])
            members = tar_video_members(archive)
            available_sequences = sorted(
                sequence
                for sequence in sequence_rows
                if locate_tar_member(members, sequence) is not None
            )
            chosen_sequences = evenly_spaced(
                available_sequences, args.videos_per_scene
            )
            if not chosen_sequences:
                raise RuntimeError(
                    f"No label sequence in {scene.label_path.name} matched an MP4 member"
                )
            for video_ordinal, sequence in enumerate(chosen_sequences, start=1):
                member = locate_tar_member(members, sequence)
                if member is None:
                    continue
                video_path = args.scratch_dir / "active_bedlam_video.mp4"
                extract_member(archive, member, video_path)
                try:
                    samples, sample_stats = samples_from_video(
                        scene_labels,
                        sequence_rows[sequence],
                        video_path,
                        pose_model,
                        device,
                        args.frames_per_video,
                        args.yolo_batch_size,
                        rotate_clockwise="closeup" in scene.label_scene.lower(),
                    )
                finally:
                    video_path.unlink(missing_ok=True)
                for key, value in sample_stats.items():
                    scene_stats[key] += float(value)
                if not samples:
                    continue
                set_bedlam_modes(student, initializer, network)
                random.shuffle(samples)
                frame_metrics, frame_steps = optimize_batches(
                    chunks(samples, args.bedlam_frame_batch_size),
                    lambda batch: bedlam_frame_loss(
                        student,
                        initializer,
                        teacher_student,
                        teacher_initializer,
                        batch,
                        device,
                    ),
                    optimizer,
                    scaler,
                    trainable,
                    f"bedlam_frame:{scene_ordinal}/{len(selected_scenes)}",
                    args.log_every,
                )
                clips = build_temporal_clips(
                    samples,
                    args.bedlam_clip_length,
                    args.bedlam_clip_stride,
                    args.clips_per_video,
                )
                clip_metrics, clip_steps = optimize_batches(
                    chunks(clips, args.bedlam_batch_size),
                    lambda batch: bedlam_clip_loss(
                        student,
                        initializer,
                        network,
                        teacher_student,
                        teacher_initializer,
                        batch,
                        device,
                    ),
                    optimizer,
                    scaler,
                    trainable,
                    f"bedlam_temporal:{scene_ordinal}/{len(selected_scenes)}",
                    args.log_every,
                )
                for prefix, metrics, steps in (
                    ("frame", frame_metrics, frame_steps),
                    ("temporal", clip_metrics, clip_steps),
                ):
                    for key, value in metrics.items():
                        scene_losses[f"{prefix}_{key}"] += value * steps
                    scene_stats[f"{prefix}_steps"] += steps
                scene_steps += frame_steps + clip_steps
                print(
                    json.dumps(
                        {
                            "scene": scene.label_scene,
                            "video": f"{video_ordinal}/{len(chosen_sequences)}",
                            "sequence": sequence,
                            "samples": len(samples),
                            "clips": len(clips),
                        }
                    ),
                    flush=True,
                )
        shutil.rmtree(scene_cache)
        if scene_stats["accepted_person_frames"] < 1:
            raise RuntimeError(
                f"YOLO26 produced no matched BEDLAM person samples in {scene.label_scene}"
            )
        validation = deployment.validation_metrics(
            student,
            initializer,
            network,
            val_labels,
            val_cache,
            args.feature_batch_size,
            device,
        )
        selected = float(validation["selection_score"]) < best_score
        if selected:
            best_score = float(validation["selection_score"])
            best_validation = validation
            best_stage = f"bedlam_scene_{scene_ordinal}"
            best_epoch = global_epoch
            save_candidate(
                source,
                student,
                initializer,
                network,
                best_path,
                best_epoch,
                best_stage,
                best_validation,
                metadata,
            )
        row = {
            "phase": "bedlam",
            "epoch": global_epoch,
            "scene": scene.label_scene,
            "attempted_person_frames": int(scene_stats["attempted_person_frames"]),
            "accepted_person_frames": int(scene_stats["accepted_person_frames"]),
            "detection_rate": scene_stats["accepted_person_frames"]
            / max(scene_stats["attempted_person_frames"], 1.0),
            "frame_steps": int(scene_stats["frame_steps"]),
            "temporal_steps": int(scene_stats["temporal_steps"]),
            **{
                f"train_{key}": value
                / max(
                    scene_stats[
                        "frame_steps" if key.startswith("frame_") else "temporal_steps"
                    ],
                    1.0,
                )
                for key, value in scene_losses.items()
            },
            **{
                f"val_{key}": value
                for key, value in compact_validation(validation).items()
            },
            "selected": selected,
            "minutes": (time.perf_counter() - scene_started) / 60.0,
        }
        history.append(row)
        write_history(args.output_dir / "bedlam_training_history.csv", history)
        bedlam_manifest.append(
            {
                "scene": scene.label_scene,
                "label": scene.label_path.name,
                "remote": scene.remote_path,
                "remote_size": scene.remote_size,
                **row,
            }
        )
        print(json.dumps(row, indent=2), flush=True)

    # Continue calibration from the best validation-selected state observed so
    # far, not blindly from the last synthetic scene.
    del student, initializer, network, teacher_student, teacher_initializer
    torch.cuda.empty_cache()
    student, initializer, network = load_selected_models(
        best_path, args.wham_repo, args.wham_checkpoint, device
    )

    print("Caching actual YOLO26 observations for 3DPW calibration...", flush=True)
    train_paths, train_observations = deployment.cache_training_observations(
        args.cache_dir / "train_yolo26_observations.pth",
        train_labels,
        train_tracks,
        image_root,
        pose_model,
        args.yolo_batch_size,
        device,
        {
            "schema": 2,
            "split": "3dpw_train_target_matched_yolo26",
            "parsed_sha256": sha256_file(args.train_parsed),
            "yolo_sha256": deployment.YOLO26_SHA256,
            "confidence_threshold": 0.5,
            "minimum_iou": 0.2,
        },
    )
    del pose_model
    torch.cuda.empty_cache()
    calibration_dataset = deployment.DeploymentClipDataset(
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
    loader_options: dict[str, Any] = {
        "batch_size": args.three_dpw_batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(calibration_dataset, **loader_options)
    stages = (
        ("joint", args.three_dpw_joint_epochs),
        ("last_stage", args.three_dpw_last_stage_epochs),
    )
    for stage, stage_epochs in stages:
        if stage_epochs <= 0:
            continue
        groups = deployment.configure_trainable(student, initializer, network, stage)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        total_steps = max(len(loader) * stage_epochs, 1)
        warmup = min(len(loader), max(total_steps // 10, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_steps, warm=warmup: deployment.cosine_schedule(
                step, total, warm
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for _ in range(stage_epochs):
            global_epoch += 1
            epoch_started = time.perf_counter()
            training = deployment.train_epoch(
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
            validation = deployment.validation_metrics(
                student,
                initializer,
                network,
                val_labels,
                val_cache,
                args.feature_batch_size,
                device,
            )
            selected = float(validation["selection_score"]) < best_score
            if selected:
                best_score = float(validation["selection_score"])
                best_validation = validation
                best_stage = f"3dpw_{stage}"
                best_epoch = global_epoch
                save_candidate(
                    source,
                    student,
                    initializer,
                    network,
                    best_path,
                    best_epoch,
                    best_stage,
                    best_validation,
                    metadata,
                )
            row = {
                "phase": "3dpw_calibration",
                "epoch": global_epoch,
                "scene": stage,
                **{f"train_{key}": value for key, value in training.items()},
                **{
                    f"val_{key}": value
                    for key, value in compact_validation(validation).items()
                },
                "selected": selected,
                "minutes": (time.perf_counter() - epoch_started) / 60.0,
            }
            history.append(row)
            write_history(args.output_dir / "bedlam_training_history.csv", history)
            print(json.dumps(row, indent=2), flush=True)

    report = {
        "schema_version": 1,
        "experiment": "bedlam_adaptation_plus_3dpw_validation_locked_calibration",
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_selection_score": best_score,
        "best_validation": best_validation,
        "baseline_validation": baseline_validation,
        "validation_delta": {
            key: float(best_validation[key]) - float(baseline_validation[key])
            for key in (
                "selection_score",
                "pose_error_deg",
                "root_error_deg",
                "body_error_deg",
                "shape_rmse",
            )
        },
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": sha256_file(best_path),
        "source_checkpoint_sha256": source_sha,
        "preflight": preflight,
        "bedlam_manifest": bedlam_manifest,
        "three_dpw_calibration_clips": len(calibration_dataset),
        "selection_policy": (
            "source checkpoint is the baseline; replacements require a lower "
            "3DPW validation score; 3DPW test is unavailable to training"
        ),
        "scientific_caveat": (
            "A prior 3DPW test result motivated this retraining. The final test is "
            "therefore confirmatory, not a pristine first-look benchmark."
        ),
        "next_action": "run_the_single_locked_3dpw_test_in_the_following_notebook_cell",
    }
    report_path = args.output_dir / "bedlam_training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "bedlam_download_manifest.json").write_text(
        json.dumps(bedlam_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_stage": best_stage,
                "best_epoch": best_epoch,
                "baseline_validation": compact_validation(baseline_validation),
                "best_validation": compact_validation(best_validation),
                "checkpoint": str(best_path),
                "checkpoint_sha256": report["best_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
