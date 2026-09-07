"""Export the distilled FastViT with the exact training RGB normalization."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class FastViTStudent(nn.Module):
    def __init__(self, model_name="fastvit_sa24", target_dim=1024):
        super().__init__()
        import timm
        # All parameters come from the student's checkpoint; no download needed.
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, target_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class RGBImageStudent(nn.Module):
    """Input: RGB pixels in [0,255]. Do not normalize them again in Swift."""
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pixels):
        return self.student((pixels / 255.0 - self.mean) / self.std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("fastvit_student_1024.pth"))
    parser.add_argument("--out", type=Path, default=Path("FastViT_WHAM_1024.mlpackage"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    import coremltools as ct
    import numpy as np

    student = FastViTStudent().cpu().eval()
    student.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = RGBImageStudent(student).eval()
    pixels = torch.rand(1, 3, 256, 256) * 255.0
    with torch.no_grad():
        traced = torch.jit.trace(model, pixels)
    mlmodel = ct.convert(
        traced,
        # Identity image preprocessing: the graph performs channel-specific normalization.
        inputs=[ct.ImageType(shape=pixels.shape, name="image_input", scale=1.0,
                             bias=[0.0, 0.0, 0.0], color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="features_1024", dtype=np.float32)],
        convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.user_defined_metadata["wham.rgb_normalization"] = "imagenet-rgb-0-255-v1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))


if __name__ == "__main__":
    main()
