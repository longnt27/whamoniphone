#!/usr/bin/env python3
"""Validate the immutable evidence selected for the mobile WHAM report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "evaluation" / "results" / "selected"
MANIFEST_PATH = EVIDENCE_ROOT / "manifest.json"

REQUIRED_ARTIFACTS = {
    "yolo26_grid_validation.json",
    "yolo26_grid_validation.csv",
    "yolo26_grid_final_3dpw.json",
    "yolo26_grid_final_3dpw.csv",
    "selected_deployment_artifact.json",
    "hmr2s_temporal_smoothing_3dpw.json",
    "hmr2s_temporal_smoothing_3dpw.csv",
    "selected_mobile_pipeline_device_benchmark.json",
    "hmr2s_coreml_export_report.json",
    "hmr2s_token_adapter_export_report.json",
    "wham_world_step_export_report.json",
}

SELECTED_YOLO = "yolo26m-pose"
SELECTED_ADAPTER_SHA256 = (
    "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"
)
SELECTED_YOLO_SHA256 = (
    "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4"
)
HMR2S_SHA256 = (
    "823728e846c901c75edb12d469fa240e07606a24cfd44c208244a94bb26fc423"
)
WHAM_SHA256 = (
    "2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def close(actual: Any, expected: float, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def validate() -> list[str]:
    errors: list[str] = []
    if not MANIFEST_PATH.is_file():
        return [f"missing {MANIFEST_PATH.relative_to(REPOSITORY_ROOT)}"]

    try:
        manifest = load_json(MANIFEST_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid manifest: {error}"]

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return ["manifest.artifacts must be an object keyed by filename"]

    missing_entries = sorted(REQUIRED_ARTIFACTS - artifacts.keys())
    if missing_entries:
        errors.append("manifest is missing: " + ", ".join(missing_entries))

    for filename in sorted(REQUIRED_ARTIFACTS):
        path = EVIDENCE_ROOT / filename
        if not path.is_file():
            errors.append(f"missing evidence file: {filename}")
            continue
        metadata = artifacts.get(filename)
        expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected_hash, str):
            errors.append(f"manifest hash missing for {filename}")
        elif sha256(path) != expected_hash:
            errors.append(f"SHA-256 mismatch for {filename}")

    if errors:
        return errors

    try:
        validation = load_json(EVIDENCE_ROOT / "yolo26_grid_validation.json")
        final = load_json(EVIDENCE_ROOT / "yolo26_grid_final_3dpw.json")
        selected = load_json(EVIDENCE_ROOT / "selected_deployment_artifact.json")
        smoothing = load_json(EVIDENCE_ROOT / "hmr2s_temporal_smoothing_3dpw.json")
        device = load_json(
            EVIDENCE_ROOT / "selected_mobile_pipeline_device_benchmark.json"
        )
        hmr_export = load_json(EVIDENCE_ROOT / "hmr2s_coreml_export_report.json")
        adapter_export = load_json(
            EVIDENCE_ROOT / "hmr2s_token_adapter_export_report.json"
        )
        world_export = load_json(EVIDENCE_ROOT / "wham_world_step_export_report.json")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return [f"could not load selected reports: {error}"]

    checks: list[tuple[bool, str]] = []
    locked = nested(validation, "locked_winner")
    final_locked = nested(final, "grid_selection", "locked_winner")
    checks.extend(
        [
            (locked == final_locked, "validation winner differs from locked test winner"),
            (locked.get("yolo_variant") == SELECTED_YOLO, "grid winner is not YOLO26m"),
            (
                locked.get("candidate") == "adapter_hmr2s_released_wham",
                "grid winner is not the residual adapter",
            ),
            (
                selected.get("yolo_variant") == SELECTED_YOLO,
                "deployment artifact has the wrong YOLO variant",
            ),
            (
                selected.get("checkpoint_sha256") == SELECTED_ADAPTER_SHA256,
                "deployment artifact has the wrong adapter hash",
            ),
            (
                selected.get("official_yolo_sha256") == SELECTED_YOLO_SHA256,
                "deployment artifact has the wrong YOLO hash",
            ),
            (
                nested(final, "grid_selection", "test_opened_after_lock") is True,
                "test split was not documented as opened after selection",
            ),
            (
                nested(smoothing, "locked_filter") == "light",
                "the selected temporal filter is not light",
            ),
            (
                close(nested(smoothing, "test", "variants", "light", "parameters", "pose_alpha"), 0.75),
                "pose smoothing alpha is not 0.75",
            ),
            (
                close(nested(smoothing, "test", "variants", "light", "parameters", "shape_alpha"), 0.35),
                "shape smoothing alpha is not 0.35",
            ),
            (
                nested(smoothing, "decision", "worth_smoothing") is True,
                "smoothing report did not select smoothing",
            ),
            (
                device.get("poseDetector") == "YOLO26m-pose",
                "device report used the wrong detector",
            ),
            (
                device.get("imageFeatureConnectionPassed") is True,
                "device image-feature connectivity guard failed",
            ),
            (
                device.get("outputSmoothingConnectionPassed") is True,
                "device smoothing connectivity guard failed",
            ),
            (
                device.get("processedSourceFrames") == 40,
                "device report is not the locked 5 x 8 frame run",
            ),
            (
                device.get("personDetections") == 40,
                "device report did not detect all 40 source frames",
            ),
            (
                nested(device, "modelProvenance", "token_adapter_checkpoint_sha256")
                == SELECTED_ADAPTER_SHA256,
                "device adapter provenance does not match selection",
            ),
            (
                hmr_export.get("checkpoint_sha256") == HMR2S_SHA256,
                "HMR2-S export used the wrong checkpoint",
            ),
            (
                adapter_export.get("checkpoint_sha256") == SELECTED_ADAPTER_SHA256,
                "Core ML adapter export used the wrong checkpoint",
            ),
            (
                world_export.get("wham_checkpoint_sha256") == WHAM_SHA256,
                "world-step export used the wrong WHAM checkpoint",
            ),
        ]
    )
    errors.extend(message for passed, message in checks if not passed)
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Final evidence validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Final evidence validation passed ({len(REQUIRED_ARTIFACTS)} artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
