from __future__ import annotations

import torch
from torch import nn

from crackmeanflow.sit import Block, RTEmbedder


def _b(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(-1, 1, 1, 1)


def _sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("1D sin-cos embedding dimension must be even")
    omega = torch.arange(dim // 2, dtype=torch.float64)
    omega = 1.0 / (10000.0 ** (omega / (dim / 2.0)))
    phase = positions.reshape(-1).double()[:, None] * omega[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1).float()


def _fixed_2d_sincos(dim: int, grid: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D sin-cos embedding dimension must be divisible by 4")
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    emb_y = _sincos_1d(dim // 2, yy.reshape(-1))
    emb_x = _sincos_1d(dim // 2, xx.reshape(-1))
    return torch.cat([emb_y, emb_x], dim=1).unsqueeze(0)


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
    ):
        super().__init__()
        img_size, patch, dim, depth, heads = map(int, (img_size, patch, dim, depth, heads))
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if depth < 1 or heads < 1:
            raise ValueError("depth and heads must be positive")
        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.grid = img_size // patch
        n_tokens = self.grid**2

        self.state_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.image_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(_fixed_2d_sincos(dim, self.grid), requires_grad=False)
        self.rt = RTEmbedder(dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)

        # Canonical MeanFlow/iMF parameterization: direct average-velocity (u)
        # and instantaneous-velocity (v) heads. Both are zero-initialized as in
        # DiT/SiT-style final layers; v is auxiliary for iMF and unused at inference.
        self.u_head = nn.Linear(dim, patch * patch)
        self.v_head = nn.Linear(dim, patch * patch)
        self._initialize_weights()

    def _initialize_weights(self):
        # Match the core DiT/SiT initialization policy where applicable.
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic_init)
        for conv in (self.state_embed, self.image_embed):
            nn.init.xavier_uniform_(conv.weight.view(conv.weight.shape[0], -1))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for layer in self.rt.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        for head in (self.u_head, self.v_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

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

    def flow_outputs(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, image: torch.Tensor):
        h = self._features(z, t, r, image)
        return {
            "u": self._unpatch(self.u_head(h)),
            "v": self._unpatch(self.v_head(h)),
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
    )
