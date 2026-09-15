#!/usr/bin/env python3
"""Export the validation-selected HMR2-S -> HMR2a token adapter to Core ML.

The checkpoint is not trained here.  This script verifies the exact artifact
selected by the YOLO26 grid, loads it strictly, converts only the residual
adapter, and checks Core ML/PyTorch numerical parity on macOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ADAPTER_SHA256 = "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TokenAdapter(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.in_projection = nn.Linear(1024, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, 1024)
        self.register_buffer("target_mean", torch.zeros(1024))
        self.register_buffer("target_std", torch.ones(1024))

    def forward(self, hmr2s_token: torch.Tensor) -> torch.Tensor:
        correction = self.out_projection(
            F.gelu(self.in_projection(self.norm(hmr2s_token)))
        )
        return hmr2s_token + correction


def load_checkpoint(path: Path) -> tuple[TokenAdapter, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload.get("uses_learned_adapter"):
        raise RuntimeError("Checkpoint is not a learned token-adapter artifact")
    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no adapter_state_dict")
    weight = state.get("in_projection.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Cannot infer adapter hidden dimension")
    model = TokenAdapter(hidden_dim=int(weight.shape[0])).eval()
    model.load_state_dict(state, strict=True)
    return model, payload


def replace_package(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    actual_sha = sha256(args.checkpoint)
    if actual_sha != ADAPTER_SHA256:
        raise RuntimeError(
            f"Adapter SHA-256 mismatch: expected {ADAPTER_SHA256}, got {actual_sha}"
        )

    torch.manual_seed(20260915)
    adapter, payload = load_checkpoint(args.checkpoint)
    example = torch.randn(1, 1, 1024, dtype=torch.float32)
    with torch.no_grad():
        reference = adapter(example).numpy()
    traced = torch.jit.trace(adapter, example, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="hmr2s_token", shape=example.shape, dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name="wham_image_token", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "WHAM-iOS packaging"
    converted.license = "Use subject to the source HMR2-S, HMR2a, and WHAM terms"
    converted.short_description = (
        "Validation-selected HMR2-S to HMR2a 1024-D residual token adapter"
    )
    converted.user_defined_metadata.update(
        {
            "checkpoint_sha256": actual_sha,
            "selection": "YOLO26m-pose 3DPW validation winner",
            "source_dimension": "1024",
            "target_dimension": "1024",
            "hidden_dimension": str(adapter.in_projection.out_features),
            "training_performed_during_export": "false",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replace_package(args.output)
    converted.save(str(args.output))

    relative_max_error = None
    if platform.system() == "Darwin":
        coreml = ct.models.MLModel(
            str(args.output), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        prediction = np.asarray(
            coreml.predict({"hmr2s_token": example.numpy().astype(np.float16)})[
                "wham_image_token"
            ]
        ).astype(np.float32)
        scale = max(float(np.max(np.abs(reference))), 1e-6)
        relative_max_error = float(np.max(np.abs(prediction - reference))) / scale
        if not np.isfinite(relative_max_error) or relative_max_error > 0.02:
            raise RuntimeError(
                "Core ML adapter parity failed: "
                f"relative max error {relative_max_error}"
            )

    report = {
        "schema_version": 1,
        "operation": "conversion_only_no_training",
        "checkpoint_sha256": actual_sha,
        "hidden_dimension": adapter.in_projection.out_features,
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "coreml_relative_max_error": relative_max_error,
        "output": str(args.output.resolve()),
        "checkpoint_metadata": {
            key: payload.get(key)
            for key in (
                "uses_learned_adapter",
                "selection_metric",
                "selected_epoch",
                "yolo_variant",
            )
            if key in payload
        },
    }
    report_path = args.report or args.output.parent / "hmr2s_token_adapter_export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
