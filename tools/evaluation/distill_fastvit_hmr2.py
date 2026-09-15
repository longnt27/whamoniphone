#!/usr/bin/env python3
"""Train a mobile FastViT student to reproduce WHAM's HMR2 image token.

This script is designed for a Kaggle GPU notebook, but it is ordinary Python so
the training recipe can be reviewed, versioned, and resumed outside Kaggle.
It fixes the original notebook's principal problems:

* both networks receive the same annotation-derived person crop;
* the student sees the same 256x192 center view used by HMR2;
* a small spatial head replaces global average pooling;
* teacher tokens are cached, making interrupted runs resumable;
* training is staged, with the pretrained backbone frozen first;
* a held-out COCO split and downstream HMR2 pose readout gate every artifact.

The resulting checkpoint is consumed by ``export_fastvit_normalized.py`` on a
Mac. Core ML prediction itself is unavailable on Kaggle's Linux workers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
import types
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
HMR2_GDRIVE_ID = "1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y"
COCO_URLS = {
    "train2017.zip": "https://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).reshape(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).reshape(3, 1, 1)
ARCHITECTURE = "fastvit_sa24_spatial_hmr2_v2"


@dataclass(frozen=True)
class PersonRecord:
    image: str
    annotation_id: int
    center_x: float
    center_y: float
    side: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_wham(work_dir: Path, provided: Path | None) -> Path:
    if provided:
        repo = provided.resolve()
    else:
        repo = work_dir / "WHAM"
        if not repo.exists():
            run(
                [
                    "git",
                    "clone",
                    "https://github.com/yohanshin/WHAM.git",
                    str(repo),
                ]
            )
        commit_present = subprocess.run(
            ["git", "cat-file", "-e", f"{WHAM_COMMIT}^{{commit}}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not commit_present:
            run(["git", "fetch", "origin", "main"], cwd=repo)
        run(["git", "checkout", "--detach", WHAM_COMMIT], cwd=repo)
    if not (repo / "lib/models/preproc/backbone/hmr2.py").exists():
        raise FileNotFoundError(f"Not a WHAM checkout: {repo}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    expected = subprocess.check_output(
        ["git", "rev-parse", WHAM_COMMIT], cwd=repo, text=True
    ).strip()
    if revision != expected:
        raise RuntimeError(
            f"WHAM checkout must be pinned to {WHAM_COMMIT}, got {revision}"
        )
    return repo


def prepare_hmr2_checkpoint(work_dir: Path, provided: Path | None) -> Path:
    if provided:
        checkpoint = provided.resolve()
    else:
        checkpoint = work_dir / "checkpoints/hmr2a.ckpt"
        if not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            import gdown

            result = gdown.download(
                id=HMR2_GDRIVE_ID, output=str(checkpoint), quiet=False
            )
            if not result:
                raise RuntimeError("HMR2 checkpoint download failed")
    if checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"HMR2 checkpoint is unexpectedly small: {checkpoint}")
    return checkpoint


def download_coco(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in COCO_URLS.items():
        archive = destination / name
        if not archive.exists():
            print(f"Downloading {url} -> {archive}")
            urllib.request.urlretrieve(url, archive)
        marker = destination / f".{name}.extracted"
        if not marker.exists():
            print(f"Extracting {archive}")
            with zipfile.ZipFile(archive) as source:
                source.extractall(destination)
            marker.touch()


def find_annotation(search_root: Path, split: str) -> Path:
    # Official COCO 2017 archives use ``train2017``/``val2017`` in both the
    # annotation and image names.  Retain the shorter aliases for older mirrors.
    names = (
        f"person_keypoints_{split}2017.json",
        f"person_keypoints_{split}.json",
    )
    dataset_roots = (
        search_root,
        search_root / "coco2017",
    )
    matches = []
    for name in names:
        matches = sorted(
            {
                path.resolve()
                for root in dataset_roots
                for path in (root / "annotations" / name, root / name)
                if path.is_file()
            }
        )
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f"Could not find one of {names} in a standard COCO layout below "
            f"{search_root}. "
            "Set --coco-root to the dataset directory containing coco2017 or "
            "annotations."
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Found multiple {name} files below {search_root}:\n  {formatted}\n"
            "Set --coco-root to one exact attached dataset so annotations and "
            "images cannot be mixed."
        )
    return matches[0]


def find_image_dir(search_root: Path, split: str) -> Path:
    dataset_roots = (
        search_root,
        search_root / "coco2017",
    )
    matches = []
    for directory_name in (f"{split}2017", split):
        matches = sorted(
            {
                path.resolve()
                for root in dataset_roots
                for path in (
                    root / directory_name,
                    root / "images" / directory_name,
                )
                if path.is_dir()
            }
        )
        if matches:
            break
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Found multiple {split} image directories below {search_root}:\n  "
            f"{formatted}\n"
            "Set --coco-root to one exact attached dataset."
        )
    raise FileNotFoundError(
        f"Could not find a {split} image directory in a standard COCO layout "
        f"below {search_root}"
    )


def discover_split(search_root: Path, split: str) -> tuple[Path, Path]:
    annotation = find_annotation(search_root, split)
    images = find_image_dir(search_root, split)
    return images, annotation


def record_from_annotation(
    annotation: dict[str, Any], image_name: str
) -> PersonRecord | None:
    if annotation.get("iscrowd", 0):
        return None
    keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float32).reshape(
        -1, 3
    )
    visible = keypoints[:, 2] > 0 if keypoints.size else np.zeros(0, dtype=bool)
    if int(visible.sum()) >= 7:
        points = keypoints[visible, :2]
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2.0
        side = float(np.max(maximum - minimum) * 1.2)
    else:
        x, y, width, height = map(float, annotation["bbox"])
        center = np.asarray((x + width / 2.0, y + height / 2.0), dtype=np.float32)
        side = max(width, height) * 1.05
    if not np.isfinite(center).all() or not math.isfinite(side) or side < 48.0:
        return None
    return PersonRecord(
        image=image_name,
        annotation_id=int(annotation["id"]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        side=float(side),
    )


def build_records(annotation_path: Path, limit: int, seed: int) -> list[PersonRecord]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_names = {int(item["id"]): item["file_name"] for item in payload["images"]}
    records = [
        record
        for annotation in payload["annotations"]
        if (
            record := record_from_annotation(
                annotation, image_names[int(annotation["image_id"])]
            )
        )
        is not None
    ]
    random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No usable person annotations in {annotation_path}")
    return records


def validate_record_images(
    image_dir: Path, records: list[PersonRecord], sample_size: int = 32
) -> None:
    missing = [
        record.image
        for record in records[:sample_size]
        if not (image_dir / record.image).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"The annotation/image roots do not match: {len(missing)} sampled files "
            f"were absent from {image_dir}, including {missing[0]}"
        )


def square_person_crop(
    image: Image.Image, record: PersonRecord, size: int = 256
) -> Image.Image:
    """Square crop with black padding, matching WHAM rather than clamping edges."""

    left = record.center_x - record.side / 2.0
    top = record.center_y - record.side / 2.0
    scale = record.side / float(size)
    return image.transform(
        (size, size),
        Image.Transform.AFFINE,
        (scale, 0.0, left, 0.0, scale, top),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (torch.from_numpy(array) - IMAGENET_MEAN) / IMAGENET_STD


class PersonCropDataset(Dataset):
    def __init__(self, image_dir: Path, records: list[PersonRecord]) -> None:
        self.image_dir = image_dir
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.image_dir / record.image) as source:
            crop = square_person_crop(source.convert("RGB"), record)
        return image_to_tensor(crop), index


class HMR2TokenEncoder(nn.Module):
    """Only the official HMR2 modules used by ``model(image, encode=True)``."""

    def __init__(self, wham_repo: Path) -> None:
        super().__init__()
        # Import only the two HMR2 backbone files. Importing through
        # ``lib.models`` executes WHAM's package initializer, which eagerly
        # imports SMPL/loguru even though token extraction does not use them.
        backbone_root = wham_repo / "lib/models/preproc/backbone"
        required = (backbone_root / "pose_transformer.py", backbone_root / "vit.py")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"Incomplete HMR2 backbone at {backbone_root}")
        package_name = "_wham_hmr2_backbone"
        package = types.ModuleType(package_name)
        package.__path__ = [str(backbone_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        pose_transformer = importlib.import_module(f"{package_name}.pose_transformer")
        vit_module = importlib.import_module(f"{package_name}.vit")

        self.backbone = vit_module.vit()
        self.transformer = pose_transformer.TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=1280,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # This slice is present in WHAM's HMR2.forward: the ViT sees 256x192.
        feature_map = self.backbone(image[:, :, :, 32:-32])
        context = feature_map.flatten(2).transpose(1, 2)
        token = torch.zeros(
            image.shape[0], 1, 1, device=image.device, dtype=image.dtype
        )
        return self.transformer(token, context=context).squeeze(1)


def subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def load_teacher(
    wham_repo: Path, checkpoint_path: Path, device: torch.device
) -> tuple[HMR2TokenEncoder, dict[str, torch.Tensor]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["state_dict"]
    model = HMR2TokenEncoder(wham_repo).eval()
    model.backbone.load_state_dict(subset(state, "backbone."), strict=True)
    model.transformer.load_state_dict(
        subset(state, "smpl_head.transformer."), strict=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if device.type == "cuda":
        model.half()
    readout = {
        "weight": state["smpl_head.decpose.weight"].float(),
        "bias": state["smpl_head.decpose.bias"].float(),
        "mean": state["smpl_head.init_body_pose"].float(),
    }
    return model, readout


def load_pose_readout(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load only the frozen HMR2 pose readout needed by cached-token training."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["state_dict"]
    return {
        "weight": state["smpl_head.decpose.weight"].float().clone(),
        "bias": state["smpl_head.decpose.bias"].float().clone(),
        "mean": state["smpl_head.init_body_pose"].float().clone(),
    }


def records_fingerprint(records: list[PersonRecord]) -> str:
    rendered = json.dumps([asdict(record) for record in records], separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def cache_teacher_tokens(
    name: str,
    teacher: HMR2TokenEncoder | None,
    image_dir: Path,
    records: list[PersonRecord],
    cache_dir: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"{name}_hmr2_tokens.npy"
    progress_path = cache_dir / f"{name}_hmr2_tokens.json"
    fingerprint = records_fingerprint(records)
    expected = {
        "schema_version": 1,
        "records": len(records),
        "record_sha256": fingerprint,
        "shape": [len(records), 1024],
        "dtype": "float16",
    }
    completed = 0
    if feature_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if all(progress.get(key) == value for key, value in expected.items()):
            completed = int(progress.get("completed", 0))
        else:
            raise RuntimeError(
                f"Cache metadata changed for {name}. Move or delete {feature_path} and rerun."
            )
        if completed >= len(records):
            print(f"Using complete {name} teacher cache: {feature_path}")
            return feature_path
        if teacher is None:
            raise RuntimeError(
                f"The read-only {name} teacher cache is incomplete "
                f"({completed}/{len(records)} records)"
            )
        features = np.load(feature_path, mmap_mode="r+")
    else:
        if teacher is None:
            raise RuntimeError(
                f"The read-only {name} teacher cache is missing: {feature_path}"
            )
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16, shape=(len(records), 1024)
        )
        atomic_json(progress_path, {**expected, "completed": 0})

    dataset = PersonCropDataset(image_dir, records)
    remaining = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    bar = tqdm(
        loader,
        desc=f"Cache HMR2 {name}",
        total=math.ceil(len(records) / batch_size),
        initial=completed // batch_size,
    )
    for images, indices in bar:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.half()
        with torch.inference_mode():
            tokens = teacher(images).float().cpu().numpy().astype(np.float16)
        index_array = np.asarray(indices, dtype=np.int64)
        features[index_array] = tokens
        completed = int(index_array.max()) + 1
        features.flush()
        atomic_json(progress_path, {**expected, "completed": completed})
    return feature_path


def feature_statistics(
    path: Path, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(path, mmap_mode="r")
    total = np.zeros(1024, dtype=np.float64)
    total_square = np.zeros(1024, dtype=np.float64)
    count = 0
    for start in range(0, len(features), chunk_size):
        values = np.asarray(features[start : start + chunk_size], dtype=np.float32)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class FastViTHMR2Student(nn.Module):
    """FastViT with a compact spatial head and a raw HMR2-token output."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "fastvit_sa24", pretrained=pretrained, num_classes=0
        )
        channels = self.backbone.num_features
        self.spatial_head = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 3)),
            nn.Flatten(),
            nn.LayerNorm(128 * 4 * 3),
            nn.Linear(128 * 4 * 3, 1024),
        )
        self.register_buffer("target_mean", torch.zeros(1024))
        self.register_buffer("target_std", torch.ones(1024))

    def normalized_token(self, image: torch.Tensor) -> torch.Tensor:
        # Match the HMR2 field of view while retaining FastViT's spatial grid.
        feature_map = self.backbone.forward_features(image[:, :, :, 32:-32])
        return self.spatial_head(feature_map)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_token(image)
        return normalized * self.target_std + self.target_mean


class CachedTokenDataset(Dataset):
    def __init__(self, images: PersonCropDataset, feature_path: Path) -> None:
        self.images = images
        self.feature_path = feature_path
        self._features: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        image, _ = self.images[index]
        token = torch.from_numpy(
            np.asarray(self._features[index], dtype=np.float32).copy()
        )
        return image, token


class PoseReadout(nn.Module):
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.linear = nn.Linear(1024, 144)
        self.linear.weight.data.copy_(values["weight"])
        self.linear.bias.data.copy_(values["bias"])
        self.register_buffer("mean_pose", values["mean"].reshape(1, 144))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.linear(token) + self.mean_pose


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    pair = rotation.reshape(-1, 24, 2, 3)
    first = F.normalize(pair[..., 0, :], dim=-1)
    second = pair[..., 1, :]
    second = F.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def pose_error_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_matrix = rotation_6d_to_matrix(first)
    second_matrix = rotation_6d_to_matrix(second)
    relative = first_matrix.transpose(-1, -2) @ second_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def configure_stage(model: FastViTHMR2Student, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.spatial_head.parameters():
        parameter.requires_grad_(True)
    if stage == "last_stage":
        for name, parameter in model.backbone.named_parameters():
            if name.startswith(("stages.3.", "final_conv.")):
                parameter.requires_grad_(True)
    elif stage != "head":
        raise ValueError(stage)


def set_training_modes(model: FastViTHMR2Student, stage: str) -> None:
    model.train()
    model.backbone.eval()
    if stage == "last_stage":
        model.backbone.stages[3].train()
        model.backbone.final_conv.train()


def cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_epoch(
    model: FastViTHMR2Student,
    pose_readout: PoseReadout,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    stage: str,
    token_loss_weight: float,
    cosine_loss_weight: float,
    pose_loss_weight: float,
    root_pose_loss_weight: float,
) -> dict[str, float]:
    set_training_modes(model, stage)
    totals = {
        "loss": 0.0,
        "token": 0.0,
        "cosine": 0.0,
        "pose": 0.0,
        "root_pose": 0.0,
    }
    samples = 0
    bar = tqdm(loader, desc=f"Train {stage}", leave=False)
    for images, targets in bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction_z = model.normalized_token(images)
            target_z = (targets - model.target_mean) / model.target_std
            predictions = prediction_z * model.target_std + model.target_mean
            token_loss = F.mse_loss(prediction_z, target_z)
            cosine_loss = (
                1.0 - F.cosine_similarity(predictions, targets, dim=-1)
            ).mean()
            predicted_pose = pose_readout(predictions)
            target_pose = pose_readout(targets)
            pose_loss = F.mse_loss(predicted_pose, target_pose)
            root_pose_loss = F.mse_loss(predicted_pose[:, :6], target_pose[:, :6])
            loss = (
                token_loss_weight * token_loss
                + cosine_loss_weight * cosine_loss
                + pose_loss_weight * pose_loss
                + root_pose_loss_weight * root_pose_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch = len(images)
        samples += batch
        for key, value in (
            ("loss", loss),
            ("token", token_loss),
            ("cosine", cosine_loss),
            ("pose", pose_loss),
            ("root_pose", root_pose_loss),
        ):
            totals[key] += float(value.detach()) * batch
        bar.set_postfix(loss=f"{totals['loss'] / samples:.4f}")
    return {key: value / samples for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: FastViTHMR2Student,
    pose_readout: PoseReadout,
    loader: DataLoader,
    device: torch.device,
    mean_baseline: bool = False,
) -> dict[str, float]:
    model.eval()
    pose_readout.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, batch_targets in tqdm(loader, desc="Validate", leave=False):
        batch_targets = batch_targets.to(device, non_blocking=True)
        if mean_baseline:
            batch_predictions = model.target_mean.expand_as(batch_targets)
        else:
            batch_predictions = model(images.to(device, non_blocking=True))
        predictions.append(batch_predictions.float().cpu())
        targets.append(batch_targets.float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    raw_cosine = F.cosine_similarity(prediction, target, dim=-1)
    prediction_z = (prediction - model.target_mean.cpu()) / model.target_std.cpu()
    target_z = (target - model.target_mean.cpu()) / model.target_std.cpu()
    centered_cosine = F.cosine_similarity(prediction_z, target_z, dim=-1, eps=1e-6)
    pose_error = pose_error_degrees(
        pose_readout.cpu()(prediction), pose_readout.cpu()(target)
    )
    pose_readout.to(device)
    return {
        "raw_cosine_mean": float(raw_cosine.mean()),
        "raw_cosine_p05": float(torch.quantile(raw_cosine, 0.05)),
        "centered_cosine_mean": float(centered_cosine.mean()),
        "centered_cosine_p05": float(torch.quantile(centered_cosine, 0.05)),
        "raw_rmse": float(torch.sqrt(F.mse_loss(prediction, target))),
        "normalized_rmse": float(torch.sqrt(F.mse_loss(prediction_z, target_z))),
        "pose_rotation_error_deg": float(pose_error.mean()),
        "root_rotation_error_deg": float(pose_error[:, 0].mean()),
        "body_rotation_error_deg": float(pose_error[:, 1:].mean()),
    }


def checkpoint_payload(
    model: FastViTHMR2Student,
    epoch: int,
    stage: str,
    validation: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "architecture": ARCHITECTURE,
        "student_state_dict": model.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "validation": validation,
        "metadata": metadata,
    }


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_selection_score(
    validation: dict[str, float],
    baseline: dict[str, float],
    args: argparse.Namespace,
) -> float:
    """Prefer checkpoints that cross all gates, then maximize raw cosine."""

    pose_baseline_limit = 0.75 * baseline["pose_rotation_error_deg"]
    deficits = (
        max(
            0.0,
            (args.accept_raw_cosine - validation["raw_cosine_mean"])
            / args.accept_raw_cosine,
        )
        + max(
            0.0,
            (args.accept_centered_cosine - validation["centered_cosine_mean"])
            / args.accept_centered_cosine,
        )
        + max(
            0.0,
            (validation["normalized_rmse"] - args.accept_normalized_rmse)
            / args.accept_normalized_rmse,
        )
        + max(
            0.0,
            (validation["pose_rotation_error_deg"] - args.accept_pose_degrees)
            / args.accept_pose_degrees,
        )
        + max(
            0.0,
            (validation["pose_rotation_error_deg"] - pose_baseline_limit)
            / pose_baseline_limit,
        )
    )
    return (
        100.0 * deficits
        - validation["raw_cosine_mean"]
        + 0.01 * validation["normalized_rmse"]
        + 0.001 * validation["pose_rotation_error_deg"]
    )


def self_test() -> None:
    seed_everything(7)
    model = FastViTHMR2Student(pretrained=False).eval()
    model.target_mean.copy_(torch.linspace(-0.5, 0.5, 1024))
    model.target_std.copy_(torch.linspace(0.5, 1.5, 1024))
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, 1024)
    image = Image.new("RGB", (100, 80), (10, 20, 30))
    crop = square_person_crop(image, PersonRecord("x", 1, 5, 5, 60))
    assert crop.size == (256, 256)
    visible_keypoints: list[float] = []
    for index in range(17):
        visible_keypoints.extend(
            (10.0 + 10 * index, 20.0 + 5 * index, 2.0 if index < 7 else 0.0)
        )
    keypoint_record = record_from_annotation(
        {
            "id": 7,
            "iscrowd": 0,
            "bbox": [0, 0, 90, 70],
            "keypoints": visible_keypoints,
        },
        "key.jpg",
    )
    assert keypoint_record is not None
    assert abs(keypoint_record.side - 72.0) < 1e-6
    bbox_record = record_from_annotation(
        {
            "id": 8,
            "iscrowd": 0,
            "bbox": [10, 20, 60, 20],
            "keypoints": [0.0] * 51,
        },
        "bbox.jpg",
    )
    assert bbox_record is not None
    assert abs(bbox_record.side - 63.0) < 1e-6
    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).repeat(2, 24)
    assert float(pose_error_degrees(identity, identity).max()) < 0.1
    print(
        "Self-test passed: annotation boxes, crop, spatial student, target scaling, and rotation metric"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/kaggle/working/wham_fastvit_distill")
    )
    parser.add_argument("--coco-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--download-coco", action="store_true")
    parser.add_argument("--wham-repo", type=Path)
    parser.add_argument("--hmr2-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        help="optional complete cache directory, including read-only Kaggle input",
    )
    parser.add_argument("--train-limit", type=int, default=48000)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-head-lr", type=float, default=8e-5)
    parser.add_argument("--finetune-backbone-lr", type=float, default=8e-6)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.25)
    parser.add_argument("--pose-loss-weight", type=float, default=0.10)
    parser.add_argument("--root-pose-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--keep-last-checkpoint",
        action="store_true",
        help="retain the redundant last-epoch checkpoint in addition to the selected best",
    )
    parser.add_argument(
        "--make-bundle",
        action="store_true",
        help="also create a zip copy of the selected checkpoint and reports",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--accept-raw-cosine", type=float, default=0.95)
    parser.add_argument("--accept-centered-cosine", type=float, default=0.50)
    parser.add_argument("--accept-normalized-rmse", type=float, default=0.85)
    parser.add_argument("--accept-pose-degrees", type=float, default=20.0)
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="resolve and validate COCO train/val inputs, then exit before GPU setup",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    seed_everything(args.seed)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    coco_root = args.coco_root.resolve()
    if args.download_coco:
        coco_root = work_dir / "coco2017"
        download_coco(coco_root)
    if not coco_root.is_dir():
        raise FileNotFoundError(
            f"COCO root is not a directory: {coco_root}. Use the exact path shown "
            "by Kaggle's Copy file path action."
        )
    print(f"Resolving COCO paths below {coco_root} ...", flush=True)
    train_images, train_annotations = discover_split(coco_root, "train2017")
    val_images, val_annotations = discover_split(coco_root, "val2017")
    if args.inspect_data:
        print("COCO_PREFLIGHT_OK")
        print(
            json.dumps(
                {
                    "coco_root": str(coco_root),
                    "train": {
                        "annotations": str(train_annotations),
                        "images": str(train_images),
                    },
                    "val": {
                        "annotations": str(val_annotations),
                        "images": str(val_images),
                    },
                },
                indent=2,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for HMR2 teacher caching")
    device = torch.device("cuda")

    print(f"Reading person annotations from {train_annotations} ...", flush=True)
    train_records = build_records(train_annotations, args.train_limit, args.seed)
    print(f"Reading person annotations from {val_annotations} ...", flush=True)
    val_records = build_records(val_annotations, args.val_limit, args.seed + 1)
    validate_record_images(train_images, train_records)
    validate_record_images(val_images, val_records)
    print(f"COCO train: {len(train_records)} crops from {train_images}", flush=True)
    print(f"COCO val:   {len(val_records)} crops from {val_images}", flush=True)

    hmr2_checkpoint = prepare_hmr2_checkpoint(work_dir, args.hmr2_checkpoint)
    cache_dir = (
        args.teacher_cache_dir.resolve()
        if args.teacher_cache_dir
        else work_dir / "cache"
    )
    teacher: HMR2TokenEncoder | None
    if args.teacher_cache_dir:
        teacher = None
        readout_values = load_pose_readout(hmr2_checkpoint)
        print("Using supplied complete teacher cache; skipping HMR2 encoder setup")
    else:
        wham_repo = prepare_wham(work_dir, args.wham_repo)
        teacher, readout_values = load_teacher(wham_repo, hmr2_checkpoint, device)
    train_features = cache_teacher_tokens(
        "train",
        teacher,
        train_images,
        train_records,
        cache_dir,
        args.teacher_batch_size,
        args.workers,
        device,
    )
    val_features = cache_teacher_tokens(
        "val",
        teacher,
        val_images,
        val_records,
        cache_dir,
        args.teacher_batch_size,
        args.workers,
        device,
    )
    del teacher
    torch.cuda.empty_cache()

    target_mean, target_std = feature_statistics(train_features)
    model = FastViTHMR2Student(pretrained=args.resume is None)
    model.target_mean.copy_(torch.from_numpy(target_mean))
    model.target_std.copy_(torch.from_numpy(target_std))
    resume_epoch = -1
    resume_metadata: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("architecture") != ARCHITECTURE:
            raise RuntimeError(
                f"Cannot resume architecture {resume.get('architecture')!r}"
            )
        model.load_state_dict(resume["student_state_dict"], strict=True)
        resume_epoch = int(resume.get("epoch", -1))
        resume_metadata = resume.get("metadata", {})
    model.to(device)
    pose_readout = PoseReadout(readout_values).to(device).eval()

    train_dataset = CachedTokenDataset(
        PersonCropDataset(train_images, train_records), train_features
    )
    val_dataset = CachedTokenDataset(
        PersonCropDataset(val_images, val_records), val_features
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    baseline = evaluate(model, pose_readout, val_loader, device, mean_baseline=True)
    print("Mean-token baseline:", json.dumps(baseline, indent=2))

    # A resume checkpoint starts another conservative last-stage fine-tune. The
    # cache is independently resumable at batch granularity.
    stages = (
        [("last_stage", args.finetune_epochs)]
        if args.resume
        else [("head", args.head_epochs), ("last_stage", args.finetune_epochs)]
    )
    history_path = work_dir / "fastvit_hmr2_history.csv"
    history: list[dict[str, Any]] = []
    if args.resume:
        history_sources = (
            history_path,
            args.resume.resolve().parent / "fastvit_hmr2_history.csv",
        )
        for history_source in history_sources:
            if history_source.exists():
                with history_source.open(newline="", encoding="utf-8") as stream:
                    history.extend(csv.DictReader(stream))
                break
    best_path = work_dir / "fastvit_hmr2_best.pth"
    if args.resume:
        if args.resume.resolve() != best_path.resolve():
            shutil.copy2(args.resume, best_path)
        if "validation" not in resume:
            raise RuntimeError("Resume checkpoint has no validation metrics")
        best_score = model_selection_score(resume["validation"], baseline, args)
    else:
        best_score = float("inf")
    global_epoch = resume_epoch + 1
    metadata = {
        "wham_commit": WHAM_COMMIT,
        "hmr2_sha256": sha256_file(hmr2_checkpoint),
        "train_annotations_sha256": sha256_file(train_annotations),
        "val_annotations_sha256": sha256_file(val_annotations),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_records_sha256": records_fingerprint(train_records),
        "val_records_sha256": records_fingerprint(val_records),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "crop": "visible COCO keypoint square x1.2; bbox x1.05 fallback; black padding",
        "input": "RGB 256x256 ImageNet normalized; both models use center 256x192",
        "loss_weights": {
            "token": args.token_loss_weight,
            "cosine": args.cosine_loss_weight,
            "pose": args.pose_loss_weight,
            "root_pose": args.root_pose_loss_weight,
        },
        "mean_baseline": baseline,
    }
    if resume_metadata is not None:
        contract_keys = (
            "wham_commit",
            "hmr2_sha256",
            "train_annotations_sha256",
            "val_annotations_sha256",
            "train_records_sha256",
            "val_records_sha256",
        )
        mismatches = [
            key
            for key in contract_keys
            if resume_metadata.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Resume checkpoint does not match this teacher/data contract: "
                + ", ".join(mismatches)
            )

    for stage, epochs in stages:
        if epochs <= 0:
            continue
        configure_stage(model, stage)
        head_parameters = [
            p for p in model.spatial_head.parameters() if p.requires_grad
        ]
        if stage == "head":
            groups = [{"params": head_parameters, "lr": args.head_lr}]
        else:
            backbone_parameters = [
                p for p in model.backbone.parameters() if p.requires_grad
            ]
            groups = [
                {"params": head_parameters, "lr": args.finetune_head_lr},
                {"params": backbone_parameters, "lr": args.finetune_backbone_lr},
            ]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.02)
        total_steps = max(len(train_loader) * epochs, 1)
        warmup_steps = min(len(train_loader), max(total_steps // 10, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_steps, warmup=warmup_steps: cosine_warmup_lambda(
                step, total, warmup
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        for _ in range(epochs):
            started = time.perf_counter()
            train_metrics = train_epoch(
                model,
                pose_readout,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                stage,
                args.token_loss_weight,
                args.cosine_loss_weight,
                args.pose_loss_weight,
                args.root_pose_loss_weight,
            )
            validation = evaluate(model, pose_readout, val_loader, device)
            row = {
                "epoch": global_epoch,
                "stage": stage,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "minutes": (time.perf_counter() - started) / 60.0,
            }
            history.append(row)
            print(json.dumps(row, indent=2))
            torch.save(
                checkpoint_payload(model, global_epoch, stage, validation, metadata),
                work_dir / "fastvit_hmr2_last.pth",
            )
            score = model_selection_score(validation, baseline, args)
            if not math.isfinite(score):
                raise RuntimeError(
                    "Validation produced a non-finite model-selection score"
                )
            if score < best_score:
                best_score = score
                torch.save(
                    checkpoint_payload(
                        model, global_epoch, stage, validation, metadata
                    ),
                    best_path,
                )
            save_history(history_path, history)
            global_epoch += 1

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["student_state_dict"], strict=True)
    model.to(device)
    final_metrics = evaluate(model, pose_readout, val_loader, device)
    gates = {
        "raw_cosine": final_metrics["raw_cosine_mean"] >= args.accept_raw_cosine,
        "centered_cosine": final_metrics["centered_cosine_mean"]
        >= args.accept_centered_cosine,
        "normalized_rmse": final_metrics["normalized_rmse"]
        <= args.accept_normalized_rmse,
        "pose_rotation": final_metrics["pose_rotation_error_deg"]
        <= args.accept_pose_degrees,
        "beats_mean_pose": final_metrics["pose_rotation_error_deg"]
        < 0.75 * baseline["pose_rotation_error_deg"],
    }
    accepted = all(gates.values())
    thresholds = {
        "raw_cosine_mean_min": args.accept_raw_cosine,
        "centered_cosine_mean_min": args.accept_centered_cosine,
        "normalized_rmse_max": args.accept_normalized_rmse,
        "pose_rotation_error_deg_max": args.accept_pose_degrees,
        "pose_error_vs_mean_baseline_max_ratio": 0.75,
    }
    # Put the decision inside the checkpoint as well as the report. The Core ML
    # exporter refuses a v2 checkpoint unless this marker is positive.
    best["validation"] = final_metrics
    best["accepted"] = accepted
    best["gates"] = gates
    best["thresholds"] = thresholds
    torch.save(best, best_path)
    report = {
        "schema_version": 2,
        "accepted": accepted,
        "gates": gates,
        "thresholds": thresholds,
        "validation": final_metrics,
        "mean_token_baseline": baseline,
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": sha256_file(best_path),
        "metadata": metadata,
    }
    report_path = work_dir / "fastvit_hmr2_training_report.json"
    atomic_json(report_path, report)
    export_dir = work_dir / "kaggle_output"
    bundle_path = work_dir / "fastvit_hmr2_kaggle_output.zip"
    if args.make_bundle:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir()
        for artifact in (
            best_path,
            work_dir / "fastvit_hmr2_history.csv",
            report_path,
        ):
            if artifact.exists():
                shutil.copy2(artifact, export_dir / artifact.name)
        shutil.make_archive(
            str(bundle_path.with_suffix("")), "zip", root_dir=export_dir
        )
        shutil.rmtree(export_dir)
    else:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        bundle_path.unlink(missing_ok=True)
    last_path = work_dir / "fastvit_hmr2_last.pth"
    if not args.keep_last_checkpoint:
        last_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    print(f"Selected checkpoint: {best_path}")
    if args.make_bundle:
        print(f"Optional Kaggle output bundle: {bundle_path}")
    if not accepted and args.fail_on_reject:
        raise SystemExit(
            "REJECTED: validation gates failed; do not export this checkpoint"
        )


if __name__ == "__main__":
    main()
