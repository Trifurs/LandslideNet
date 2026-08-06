"""Task-adapted deep comparison networks supplied with the project."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, requested: int = 8) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
            nn.GELU(),
        )


class DBPFNetAdapter(nn.Module):
    """Dense CNN/factor-sequence LSTM adapter for DBPFNet."""

    def __init__(self, num_bands, num_classes=2):
        super().__init__()
        self.spatial = nn.Sequential(
            _ConvNormAct(num_bands, 32, 5),
            _ConvNormAct(32, 64, 3, dilation=2),
            _ConvNormAct(64, 64, 3),
        )
        self.sequence = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.sequence_norm = nn.LayerNorm(64)
        self.fusion = nn.Sequential(
            _ConvNormAct(128, 64),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, num_classes, 1),
        )
        self.fusion[-1]._landslidenet_classification_head = True

    def forward(self, x):
        spatial = self.spatial(x)
        pooled = F.avg_pool2d(x, kernel_size=16, stride=16, ceil_mode=True)
        batch, bands, height, width = pooled.shape
        sequences = pooled.permute(0, 2, 3, 1).reshape(-1, bands, 1)
        _output, (hidden, _cell) = self.sequence(sequences)
        sequence = self.sequence_norm(torch.cat((hidden[-2], hidden[-1]), dim=1))
        sequence = sequence.reshape(batch, height, width, 64).permute(0, 3, 1, 2)
        sequence = F.interpolate(
            sequence,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat((spatial, sequence), dim=1))


class _FactorAttention(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        hidden = max(8, num_bands // 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_bands, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class DALSFAdapter(nn.Module):
    """Factor-attention and multi-scale spatial-fusion adapter."""

    def __init__(self, num_bands, num_classes=2):
        super().__init__()
        self.factor_attention = _FactorAttention(num_bands)
        self.stem = _ConvNormAct(num_bands, 48)
        self.scales = nn.ModuleList(
            [
                _ConvNormAct(48, 48, 3, dilation=dilation)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.fuse = nn.Sequential(
            _ConvNormAct(48 * 4, 96, 1),
            _ConvNormAct(96, 64),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, num_classes, 1),
        )
        self.fuse[-1]._landslidenet_classification_head = True

    def forward(self, x):
        features = self.stem(self.factor_attention(x))
        return self.fuse(
            torch.cat([branch(features) for branch in self.scales], dim=1)
        )


class LGCNetAdapter(nn.Module):
    """Local CNN/global Transformer parallel feature-fusion adapter."""

    def __init__(self, num_bands, num_classes=2):
        super().__init__()
        self.local = nn.Sequential(
            _ConvNormAct(num_bands, 32),
            _ConvNormAct(32, 64),
            _ConvNormAct(64, 64),
        )
        self.patch_embed = nn.Conv2d(num_bands, 64, kernel_size=16, stride=16)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.global_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fuse = nn.Sequential(
            _ConvNormAct(128, 96, 1),
            _ConvNormAct(96, 64),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, num_classes, 1),
        )
        self.fuse[-1]._landslidenet_classification_head = True

    def forward(self, x):
        local = self.local(x)
        patches = self.patch_embed(x)
        batch, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        y, x_coord = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        frequencies = torch.arange(channels // 4, device=x.device, dtype=x.dtype)
        frequencies = torch.pow(
            10000.0,
            -frequencies / max(channels // 4 - 1, 1),
        )
        positional = torch.cat(
            (
                torch.sin(x_coord[..., None] / frequencies),
                torch.cos(x_coord[..., None] / frequencies),
                torch.sin(y[..., None] / frequencies),
                torch.cos(y[..., None] / frequencies),
            ),
            dim=-1,
        ).reshape(1, height * width, channels)
        tokens = self.global_encoder(tokens + positional)
        global_features = tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        global_features = F.interpolate(
            global_features,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat((local, global_features), dim=1))


def build_comparison_deep_model(
    name: str,
    num_bands: int,
    num_classes: int = 2,
) -> nn.Module:
    builders = {
        "dbpfnet": DBPFNetAdapter,
        "da_lsf": DALSFAdapter,
        "lgc_net": LGCNetAdapter,
    }
    try:
        builder = builders[name]
    except KeyError as error:
        raise ValueError(f"Not a registered deep comparison model: {name}") from error
    return builder(num_bands, num_classes)


__all__ = [
    "DALSFAdapter",
    "DBPFNetAdapter",
    "LGCNetAdapter",
    "build_comparison_deep_model",
]
