#!/usr/bin/env python3
"""Export the distilled FastViT with its training normalization in-graph."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn


PRODUCT_VALIDATION_THRESHOLDS = {
    "max_pose_degradation_deg": 1.15,
    "max_relative_pose_degradation": 0.10,
    "max_teacher_drift_deg": 5.0,
}


class LegacyFastViTStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "fastvit_sa24", pretrained=False, num_classes=0
        )
        self.proj = nn.Linear(self.backbone.num_features, 1024)

    def forward(self, image):
        return self.proj(self.backbone(image))


class SpatialFastViTHMR2Student(nn.Module):
    """Architecture emitted by distill_fastvit_hmr2.py (format version 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "fastvit_sa24", pretrained=False, num_classes=0
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

    def normalized_token(self, image):
        feature_map = self.backbone.forward_features(image[:, :, :, 32:-32])
        return self.spatial_head(feature_map)

    def forward(self, image):
        normalized = self.normalized_token(image)
        return normalized * self.target_std + self.target_mean


def load_student(
    path: Path,
    allow_legacy: bool = False,
    accept_product_validation: bool = False,
) -> tuple[nn.Module, str, dict[str, Any]]:
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "student_state_dict" in checkpoint:
        architecture = checkpoint.get("architecture")
        if architecture != "fastvit_sa24_spatial_hmr2_v2":
            raise RuntimeError(f"Unsupported distilled architecture: {architecture!r}")
        required_gates = {
            "raw_cosine",
            "centered_cosine",
            "normalized_rmse",
            "pose_rotation",
            "beats_mean_pose",
        }
        gates = checkpoint.get("gates", {})
        gates_pass = required_gates.issubset(gates) and all(
            gates[name] is True for name in required_gates
        )
        validation = checkpoint.get("validation", {})
        acceptance_basis = "predeclared_training_gates"
        original_accepted = checkpoint.get("accepted") is True and gates_pass
        if not original_accepted:
            validation = checkpoint.get("phase3_validation", {})
            product_gates = {
                "absolute_pose_degradation": (
                    validation.get("pose_degradation_deg", float("inf"))
                    <= PRODUCT_VALIDATION_THRESHOLDS["max_pose_degradation_deg"]
                ),
                "relative_pose_degradation": (
                    validation.get("relative_pose_degradation", float("inf"))
                    <= PRODUCT_VALIDATION_THRESHOLDS["max_relative_pose_degradation"]
                ),
                "student_teacher_pose_drift": (
                    validation.get("student_teacher_pose_drift_deg", float("inf"))
                    <= PRODUCT_VALIDATION_THRESHOLDS["max_teacher_drift_deg"]
                ),
            }
            if not accept_product_validation or not all(product_gates.values()):
                raise RuntimeError(
                    "Refusing to export an unaccepted v2 checkpoint. The original "
                    "validation gate did not pass. Use --accept-product-validation "
                    "only for a phase-three checkpoint that satisfies the explicitly "
                    "post-hoc 1.15 degree / 10% / 5 degree product policy."
                )
            acceptance_basis = "post_hoc_product_validation_2026-09-10"

        student = SpatialFastViTHMR2Student()
        student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        checkpoint["_export_acceptance"] = {
            "accepted": True,
            "basis": acceptance_basis,
            "original_validation_accepted": original_accepted,
            "deployment_accepted": False,
            "validation": validation,
            "product_thresholds": (
                PRODUCT_VALIDATION_THRESHOLDS
                if acceptance_basis.startswith("post_hoc")
                else {}
            ),
        }
        print(f"Loaded {architecture}; recorded validation: {validation}")
        return student, architecture, checkpoint

    if not allow_legacy:
        raise RuntimeError(
            "Refusing the legacy global-pool checkpoint because it failed HMR2 "
            "feature validation. Use --allow-legacy only to reproduce diagnostics."
        )
    student = LegacyFastViTStudent()
    student.load_state_dict(checkpoint, strict=True)
    return student, "legacy_fastvit_sa24_global_pool_v1", {}


class ByteImageNormalized(nn.Module):
    """Accept Core ML's 0...255 RGB tensor and reproduce training transforms."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def forward(self, image):
        normalized = (image / 255.0 - self.mean) / self.std
        return self.student(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument(
        "--accept-product-validation",
        action="store_true",
        help=(
            "Allow the explicitly post-hoc product policy: <=1.15 degree absolute "
            "degradation, <=10%% relative degradation, and <=5 degree teacher drift. "
            "This does not mark deployment accepted; the untouched test is still required."
        ),
    )
    parser.add_argument(
        "--full-evaluation-report",
        type=Path,
        help=(
            "Optional full 3DPW tradeoff JSON. Its controlled FastViT result is "
            "embedded as diagnostic provenance; it never marks deployment accepted."
        ),
    )
    args = parser.parse_args()

    student, architecture, checkpoint = load_student(
        args.weights,
        allow_legacy=args.allow_legacy,
        accept_product_validation=args.accept_product_validation,
    )
    student = student.cpu().eval()
    model = ByteImageNormalized(student).cpu().eval()

    example = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)
    traced = torch.jit.trace(model, example, check_trace=True)
    converted = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image_input",
                shape=example.shape,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="features_1024")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "WHAM-iOS"
    converted.short_description = (
        f"HMR2-token student ({architecture}) with in-graph normalization"
    )
    acceptance = checkpoint.get("_export_acceptance", {})
    validation = acceptance.get("validation", checkpoint.get("validation", {}))
    product_thresholds = acceptance.get("product_thresholds", {})
    full_evaluation: dict[str, Any] = {}
    if args.full_evaluation_report is not None:
        full_evaluation = json.loads(
            args.full_evaluation_report.read_text(encoding="utf-8")
        )
    full_drift = (
        full_evaluation.get("controlled_fastvit", {})
        .get("student_teacher_pose_drift", {})
        .get("all_joints_deg", {})
        .get("mean")
    )
    full_test_status = "not_supplied"
    if full_drift is not None:
        full_test_status = (
            "pass"
            if float(full_drift)
            <= PRODUCT_VALIDATION_THRESHOLDS["max_teacher_drift_deg"]
            else "failed_guardrail"
        )
    converted.user_defined_metadata.update(
        {
            "wham_student_architecture": architecture,
            "wham_hmr2_validation_accepted": str(
                acceptance.get("accepted", checkpoint.get("accepted") is True)
            ).lower(),
            "wham_hmr2_acceptance_basis": str(acceptance.get("basis", "legacy")),
            "wham_hmr2_original_validation_accepted": str(
                acceptance.get("original_validation_accepted", False)
            ).lower(),
            "wham_hmr2_deployment_accepted": str(
                acceptance.get("deployment_accepted", False)
            ).lower(),
            "wham_hmr2_raw_cosine_mean": str(validation.get("raw_cosine_mean", "")),
            "wham_hmr2_centered_cosine_mean": str(
                validation.get("centered_cosine_mean", "")
            ),
            "wham_hmr2_pose_rotation_error_deg": str(
                validation.get("pose_rotation_error_deg", "")
            ),
            "wham_hmr2_pose_degradation_deg": str(
                validation.get("pose_degradation_deg", "")
            ),
            "wham_hmr2_relative_pose_degradation": str(
                validation.get("relative_pose_degradation", "")
            ),
            "wham_hmr2_student_teacher_pose_drift_deg": str(
                validation.get("student_teacher_pose_drift_deg", "")
            ),
            "wham_hmr2_product_max_pose_degradation_deg": str(
                product_thresholds.get("max_pose_degradation_deg", "")
            ),
            "wham_hmr2_full_3dpw_status": full_test_status,
            "wham_hmr2_full_3dpw_teacher_drift_deg": str(
                "" if full_drift is None else full_drift
            ),
            "wham_hmr2_full_3dpw_report": (
                ""
                if args.full_evaluation_report is None
                else args.full_evaluation_report.name
            ),
        }
    )

    output = args.output.resolve()
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))

    if platform.system() != "Darwin":
        print(
            "Saved the Core ML package. Runtime parity is skipped because "
            "MLModel.predict is only available on macOS; run this exporter once "
            "on the Mac before installing the package on iPhone."
        )
        return

    # Test the public image interface, not just the traced tensor graph.
    compiled = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
    rgb = example[0].permute(1, 2, 0).byte().numpy()
    with torch.no_grad():
        expected = model(example).numpy()
    actual = np.asarray(
        compiled.predict({"image_input": Image.fromarray(rgb)})["features_1024"]
    )
    max_error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or max_error > 0.15:
        raise RuntimeError(f"Core ML parity check failed (max error={max_error})")
    print(f"Saved {output}")
    print(f"Core ML parity: max |feature delta| = {max_error:.6f}")
    print(f"Smoke feature L2 norm = {np.linalg.norm(actual):.3f}")


if __name__ == "__main__":
    main()
