#!/usr/bin/env python3
"""Frozen HMR2.0-S inference wrapper used by Kaggle and Core ML export.

The weights and architecture come from the authors' TruncHierVFM release.  This
module intentionally builds only the ViTPose-S backbone and SMPL transformer
head, avoiding the training, rendering, discriminator, and optional-backbone
dependencies in the research repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

HMR2S_REPOSITORY = "https://github.com/nttcom/TruncHierVFM.git"
HMR2S_COMMIT = "d69218f411e003621f29df1940b23b076067fad1"
HMR2S_GOOGLE_DRIVE_ID = "1k6kdJXQtmOtHffGamrRuig1x3HzTvwlh"
HMR2S_CHECKPOINT_SHA256 = (
    "823728e846c901c75edb12d469fa240e07606a24cfd44c208244a94bb26fc423"
)
HMR2S_MODEL_DIRECTORY = "hmr_vit-small_d3-a4x16-m128"


def _load_official_vit_module(repository: Path) -> Any:
    source = (
        repository
        / "4D-Humans"
        / "hmr2"
        / "models"
        / "backbones"
        / "vit.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Official HMR2.0-S ViT source is missing: {source}")
    spec = importlib.util.spec_from_file_location("_official_hmr2s_vit", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreNorm(nn.Module):
    def __init__(self, dim: int, operation: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = operation

    def forward(self, value: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(value), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.0)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query, key, result = self.to_qkv(value).chunk(3, dim=-1)
        query, key, result = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads)
            for tensor in (query, key, result)
        )
        attention = self.attend(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale
        )
        output = torch.matmul(self.dropout(attention), result)
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, heads: int, dim_head: int
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.0)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        key, result = self.to_kv(context).chunk(2, dim=-1)
        query = self.to_q(value)
        query, key, result = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads)
            for tensor in (query, key, result)
        )
        attention = self.attend(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale
        )
        output = torch.matmul(self.dropout(attention), result)
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 3,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 128,
        context_dim: int = 384,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, SelfAttention(dim, heads, dim_head)),
                        PreNorm(
                            dim,
                            CrossAttention(dim, context_dim, heads, dim_head),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim)),
                    ]
                )
            )

    def forward(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        for self_attention, cross_attention, feed_forward in self.layers:
            value = self_attention(value) + value
            value = cross_attention(value, context=context) + value
            value = feed_forward(value) + value
        return value


class TransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_token_embedding = nn.Linear(1, 1024)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1024))
        self.dropout = nn.Dropout(0.0)
        self.transformer = TransformerCrossAttention()

    def forward(
        self, token: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        value = self.dropout(self.to_token_embedding(token))
        value = value + self.pos_embedding
        return self.transformer(value, context=context)


class SMPLTransformerDecoderHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TransformerDecoder()
        self.decpose = nn.Linear(1024, 144)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.register_buffer("init_body_pose", torch.zeros(1, 144))
        self.register_buffer("init_betas", torch.zeros(1, 10))
        self.register_buffer("init_cam", torch.zeros(1, 3))


def hmr2_rot6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Use the exact column-basis convention in the released HMR2.0-S head."""

    rotation = rotation.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    first = F.normalize(rotation[:, :, 0], dim=-1)
    second_raw = rotation[:, :, 1]
    second = F.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class FrozenHMR2S(nn.Module):
    """Released HMR2.0-S front end with the WHAM conditioning token exposed."""

    def __init__(self, repository: Path, checkpoint: Path) -> None:
        super().__init__()
        vit_module = _load_official_vit_module(repository)
        self.backbone = vit_module.ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.1,
            out_stage=4,
        )
        self.smpl_head = SMPLTransformerDecoderHead()

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {
            key: value
            for key, value in payload["state_dict"].items()
            if key.startswith(("backbone.", "smpl_head."))
        }
        missing, unexpected = self.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"HMR2.0-S checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.eval()

    def forward(
        self, normalized_square_crop: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The official HMR2 family receives a 256x256 square crop and discards
        # 32 pixels on each horizontal side for its 256x192 ViT input.
        context_map = self.backbone(normalized_square_crop[:, :, :, 32:-32])
        context = rearrange(context_map, "b c h w -> b (h w) c")
        token = torch.zeros(
            normalized_square_crop.shape[0],
            1,
            1,
            dtype=normalized_square_crop.dtype,
            device=normalized_square_crop.device,
        )
        token = self.smpl_head.transformer(token, context).squeeze(1)

        raw_pose = self.smpl_head.decpose(token) + self.smpl_head.init_body_pose
        betas = self.smpl_head.decshape(token) + self.smpl_head.init_betas
        camera = self.smpl_head.deccam(token) + self.smpl_head.init_cam

        # WHAM initializes from the first two rows of HMR2's orthonormalized
        # rotation matrices, matching its official HMR2a preprocessing path.
        matrices = hmr2_rot6d_to_matrix(raw_pose).reshape(-1, 24, 3, 3)
        wham_pose_6d = matrices[:, :, :2, :].reshape(-1, 24, 6)
        return token, wham_pose_6d, betas, camera


def checkpoint_smpl_buffers(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    names = (
        "shapedirs",
        "v_template",
        "J_regressor",
        "posedirs",
        "parents",
        "lbs_weights",
    )
    result = {name: state[f"smpl.{name}"].detach().float() for name in names}
    result["parents"] = state["smpl.parents"].detach().long()
    return result
