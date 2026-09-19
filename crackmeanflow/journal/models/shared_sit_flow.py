from __future__ import annotations

import math
import torch
from torch import nn

from crackmeanflow.sit import Block, RTEmbedder


def _b(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(-1, 1, 1, 1)


class SharedCrackSiT(nn.Module):
    """Direct-mask SiT backbone shared exactly by MF and iMF arms.

    The Transformer trunk and both output heads are identical for S0/S1.
    Only the training objective is allowed to differ.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch: int = 8,
        dim: int = 384,
        depth: int = 10,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        background_init: float = -0.95,
    ):
        super().__init__()
        img_size, patch, dim, depth, heads = map(int, (img_size, patch, dim, depth, heads))
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if depth < 1 or heads < 1:
            raise ValueError("depth and heads must be positive")
        if not (-0.999 < float(background_init) < 0.999):
            raise ValueError("background_init must lie inside (-0.999, 0.999)")

        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.grid = img_size // patch
        n_tokens = self.grid**2

        self.state_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.image_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.rt = RTEmbedder(dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)

        self.clean_u_head = nn.Linear(dim, patch * patch)
        self.clean_v_head = nn.Linear(dim, patch * patch)
        bias = float(math.atanh(float(background_init)))
        for head in (self.clean_u_head, self.clean_v_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, bias)

    def _unpatch(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        g, p = self.grid, self.patch
        return (
            tokens.reshape(b, g, g, p, p, 1)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, 1, g * p, g * p)
        )

    def _features(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        h = (self.state_embed(z) + self.image_embed(image)).flatten(2).transpose(1, 2) + self.pos
        c = self.rt(r, t)
        for block in self.blocks:
            h = block(h, c)
        scale, shift = self.ada_out(c).chunk(2, dim=-1)
        return self.norm(h) * (1 + scale[:, None]) + shift[:, None]

    def clean_predictions(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, image: torch.Tensor):
        h = self._features(z, t, r, image)
        clean_u = torch.tanh(self._unpatch(self.clean_u_head(h)))
        clean_v = torch.tanh(self._unpatch(self.clean_v_head(h)))
        return clean_u, clean_v

    @staticmethod
    def clean_to_velocity(z: torch.Tensor, clean: torch.Tensor, t: torch.Tensor, min_t: float = 0.05):
        return (z - clean) / _b(t.clamp(min=float(min_t)))

    def flow_outputs(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, image: torch.Tensor):
        clean_u, clean_v = self.clean_predictions(z, t, r, image)
        return {
            "u": self.clean_to_velocity(z, clean_u, t),
            "v": self.clean_to_velocity(z, clean_v, t),
            "clean_u": clean_u,
            "clean_v": clean_v,
        }

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, y=None, **kwargs):
        del kwargs
        if y is None:
            raise ValueError("SharedCrackSiT requires conditioning image y")
        return self.flow_outputs(z, t, r, y)["u"]

    def get_seg_logits(self):
        return None

    def architecture_signature(self):
        return {
            "img_size": self.img_size,
            "patch": self.patch,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
        }


def build_shared_crack_sit(cfg):
    return SharedCrackSiT(
        img_size=cfg.get("img_size", 256),
        patch=cfg.get("patch", 8),
        dim=cfg.get("dim", 384),
        depth=cfg.get("depth", 10),
        heads=cfg.get("heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        background_init=cfg.get("background_init", -0.95),
    )
