"""Clean-room compatible multi-task U-Net for the Conference CrackMeanFlow path.

This module implements the documented historical tensor contract and network
layout used by the project, without copying third-party source text.  The
forward contract is deliberately fixed as::

    velocity, segmentation_logits = model(x_t, rt_embedding, rgb_image)

The implementation is written from the architecture specification captured by
this project (dual decoders, DDPM-style residual blocks, image-conditioning
stem, and the local-attention residual feature extractor).  No external
CrackDiff/retinaDiffusion source file is required at runtime.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        gate = x.new_empty(shape).bernoulli_(keep)
        return x * gate / keep


def _group_norm(channels: int) -> nn.GroupNorm:
    if channels % 32 == 0:
        groups = 32
    elif channels % 16 == 0:
        groups = 16
    elif channels % 8 == 0:
        groups = 8
    else:
        groups = 4
    return nn.GroupNorm(groups, channels)


def _init_xavier(module: nn.Module, *, gain: float = 1.0) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _DepthwiseGroupedConv(nn.Module):
    def __init__(self, channels: int, group_width: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=channels // group_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class _SpatialMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        hidden = out_channels // 4
        self.fc1 = nn.Conv2d(in_channels, hidden, 1)
        self.gn1 = nn.GroupNorm(hidden // 4, hidden)
        self.dwconv = _DepthwiseGroupedConv(hidden)
        self.gn2 = nn.GroupNorm(hidden // 4, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, out_channels, 1)
        self.gn3 = nn.GroupNorm(out_channels // 4, out_channels)
        self.drop = nn.Dropout(dropout)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                module.weight.data.normal_(0.0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn1(self.fc1(x))
        x = self.gn2(self.dwconv(x))
        x = self.drop(self.act(x))
        x = self.drop(self.gn3(self.fc2(x)))
        return x


class _LocalAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, key_dim: int = 16, local_size: int = 7):
        super().__init__()
        if out_channels % heads:
            raise ValueError("out_channels must be divisible by heads")
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = out_channels // heads
        self.local_size = local_size
        self.padding = (local_size - 1) // 2
        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, key_dim * heads, 1, bias=False),
            nn.GroupNorm(key_dim * heads // 4, key_dim * heads),
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, key_dim, 1, bias=False),
            nn.GroupNorm(key_dim // 4, key_dim),
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.value_dim, 1, bias=False),
            nn.GroupNorm(self.value_dim // 4, self.value_dim),
        )
        self.embedding = nn.Parameter(torch.randn(key_dim, 1, 1, local_size, local_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        pixels = height * width
        q = self.queries(x).reshape(batch, self.heads, self.key_dim, pixels)
        k = self.keys(x).reshape(batch, self.key_dim, 1, pixels).softmax(dim=-1)
        v = self.values(x).reshape(batch, self.value_dim, 1, pixels)

        global_context = torch.einsum("bkum,bvum->bkv", k, v)
        global_context = torch.einsum("bhkn,bkv->bhvn", q, global_context)

        v3 = v.reshape(batch, 1, self.value_dim, height, width)
        local_context = F.conv3d(v3, self.embedding, padding=(0, self.padding, self.padding))
        local_context = local_context.reshape(batch, self.key_dim, self.value_dim, pixels)
        local_context = torch.einsum("bhkn,bkvn->bhvn", q, local_context)
        return (global_context + local_context).reshape(batch, -1, height, width)


class _TransformerFeatureBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.3, drop_path: float = 0.0):
        super().__init__()
        self.attn = _LocalAttention(channels, channels)
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = _SpatialMLP(channels, channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        return x + self.drop_path(self.mlp(x))


class _FeatureBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(in_channels, out_channels) // 4
        self.reduce = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.reduce_norm = nn.GroupNorm(hidden // 4, hidden)
        self.body = nn.Sequential(_TransformerFeatureBlock(hidden), nn.GELU())
        self.expand = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.expand_norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.activation = nn.GELU()
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(out_channels // 4, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.reduce_norm(self.reduce(x)))
        h = self.body(h)
        h = self.activation(self.expand_norm(self.expand(h)))
        return h + self.skip(x)


class _DenseConditioningBlock(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 32, beta: float = 0.2):
        super().__init__()
        self.beta = float(beta)
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.act = nn.LeakyReLU()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            _FeatureBottleneck(channels, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.act(self.conv1(x))
        b2 = self.act(self.conv2(torch.cat([b1, x], dim=1)))
        b3 = self.act(self.conv3(torch.cat([b2, b1, x], dim=1)))
        b4 = self.act(self.conv4(torch.cat([b3, b2, b1, x], dim=1)))
        enriched = x + self.context(x)
        dense = self.conv5(torch.cat([b4, b3, b2, b1, enriched], dim=1))
        return enriched + self.beta * dense


class _RRDB(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 32, beta: float = 0.2):
        super().__init__()
        self.beta = float(beta)
        self.rdb = _DenseConditioningBlock(channels, growth, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta * self.rdb(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T: int, d_model: int, out_dim: int):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        positions = torch.arange(T, dtype=torch.float32)[:, None]
        angles = positions * frequencies[None, :]
        table = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).reshape(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(table),
            nn.Linear(d_model, out_dim),
            Swish(),
            nn.Linear(out_dim, out_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_xavier(module)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.timembedding(t)


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        _init_xavier(self.main)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        del temb
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Conv2d(channels, channels, 3, padding=1)
        _init_xavier(self.main)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        del temb
        return self.main(F.interpolate(x, scale_factor=2, mode="nearest"))


class _SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        for layer in (self.q, self.k, self.v, self.proj):
            _init_xavier(layer)
        _init_xavier(self.proj, gain=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        k = self.k(h).reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * (channels ** -0.5)
        weights = weights.softmax(dim=-1)
        v = self.v(h).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        h = torch.bmm(weights, v).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return x + self.proj(h.contiguous())


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float, attention: bool = False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_proj = nn.Sequential(Swish(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attn = _SelfAttention(out_channels) if attention else nn.Identity()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                _init_xavier(module)
        _init_xavier(self.block2[-1], gain=1e-5)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.block2(h)
        return self.attn(h + self.skip(x))


class UNet(nn.Module):
    """Dual-decoder U-Net used by the Conference CrackMeanFlow baseline."""

    def __init__(self, T: int, ch: int, ch_mult: Iterable[int], attn: Iterable[int], num_res_blocks: int, dropout: float):
        super().__init__()
        multipliers = list(ch_mult)
        attention_levels = set(attn)
        if any(level < 0 or level >= len(multipliers) for level in attention_levels):
            raise ValueError("attention level index out of range")
        time_dim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, time_dim)
        self.x_head = nn.Conv2d(1, ch, 3, padding=1)
        self.seg_head = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1), _RRDB(channels=ch))

        self.downblocks = nn.ModuleList()
        skip_channels = [ch]
        current = ch
        for level, multiplier in enumerate(multipliers):
            target = ch * multiplier
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(current, target, time_dim, dropout, attention=level in attention_levels))
                current = target
                skip_channels.append(current)
            if level != len(multipliers) - 1:
                self.downblocks.append(DownSample(current))
                skip_channels.append(current)

        self.middleblocks = nn.ModuleList(
            [ResBlock(current, current, time_dim, dropout, attention=True), ResBlock(current, current, time_dim, dropout, attention=False)]
        )
        self.noisy_upblocks = nn.ModuleList()
        self.seg_upblocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(multipliers))):
            target = ch * multiplier
            for _ in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                self.noisy_upblocks.append(ResBlock(skip + current, target, time_dim, dropout, attention=level in attention_levels))
                self.seg_upblocks.append(ResBlock(skip + current + target, target, time_dim, dropout, attention=level in attention_levels))
                current = target
            if level != 0:
                self.noisy_upblocks.append(UpSample(current))
                self.seg_upblocks.append(UpSample(current))
        if skip_channels:
            raise AssertionError("internal skip-channel bookkeeping mismatch")

        self.noisy_tail = nn.Sequential(nn.GroupNorm(32, current), Swish(), nn.Conv2d(current, 1, 3, padding=1))
        self.seg_tail = nn.Sequential(nn.GroupNorm(32, current), Swish(), nn.Conv2d(current, 1, 3, padding=1))
        _init_xavier(self.x_head)
        _init_xavier(self.noisy_tail[-1], gain=1e-5)
        _init_xavier(self.seg_tail[-1], gain=1e-5)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor, entropy: torch.Tensor | None = None):
        temb = self.time_embedding(t)
        image_features = self.seg_head(image)
        if entropy is not None:
            image_features = image_features + self.seg_head(entropy)
        h = self.x_head(x_t) + image_features

        skips = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            skips.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)

        shared_bottom = h
        noisy_features = []
        skip_index = len(skips)
        for layer in self.noisy_upblocks:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips[skip_index - 1]], dim=1), temb)
                skip_index -= 1
                noisy_features.append(h)
            else:
                h = layer(h, temb)
        if skip_index != 0:
            raise AssertionError("noisy decoder did not consume all skips")

        seg_index = 0
        for layer in self.seg_upblocks:
            if isinstance(layer, ResBlock):
                shared_bottom = torch.cat([shared_bottom, skips.pop(), noisy_features[seg_index]], dim=1)
                seg_index += 1
            shared_bottom = layer(shared_bottom, temb)
        if skips:
            raise AssertionError("segmentation decoder did not consume all skips")

        return self.noisy_tail(h), self.seg_tail(shared_bottom)
