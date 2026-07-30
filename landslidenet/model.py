"""The proposed LandslideNet architecture and its controlled ablations.

Comparison models deliberately live under :mod:`Tools.model_comparisons`.
Keeping this module limited to the study model prevents a task-adapted
comparison network from being mistaken for part of LandslideNet itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def _valid_groups(channels: int, requested: int = 8) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
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


class DCSELayer(nn.Module):
    """Dynamic channel squeeze-and-excitation used by LandslideNet."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        self.phi = nn.Parameter(torch.empty(hidden, channels))
        nn.init.kaiming_uniform_(self.phi, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        theta = self.theta(x).flatten(1)
        dynamic = torch.matmul(theta, F.softmax(self.phi, dim=-1))
        return x * dynamic.sigmoid()[:, :, None, None]


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(in_channels), in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ControlledSpatialBlock(nn.Module):
    """One controlled block for the baseline/DCSE/SPB/full ablation matrix."""

    def __init__(self, in_channels, out_channels, deformable, dcse, dropout):
        super().__init__()
        self.deformable = bool(deformable)
        if self.deformable:
            self.offset = nn.Conv2d(in_channels, 18, 3, padding=1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.spatial = DeformConv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.offset = None
            self.spatial = nn.Conv2d(
                in_channels, out_channels, 3, padding=1, bias=False
            )
        self.norm = nn.GroupNorm(_valid_groups(out_channels), out_channels)
        self.channel = DCSELayer(out_channels) if dcse else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_valid_groups(out_channels), out_channels),
            )
        )

    def forward(self, x):
        if self.offset is None:
            out = self.spatial(x)
        else:
            out = self.spatial(x, self.offset(x))
        out = self.dropout(self.channel(self.norm(out)))
        return F.relu(out + self.shortcut(x), inplace=True)


class LandslideNet(nn.Module):
    """Proposed network; flags are exposed only for controlled ablations."""

    def __init__(self, num_bands, num_classes=2, *, use_spb=True, use_dcse=True):
        super().__init__()
        self.stem = ConvNormAct(num_bands, 64)
        self.enc1 = ControlledSpatialBlock(64, 128, use_spb, use_dcse, 0.10)
        self.enc2 = ControlledSpatialBlock(128, 256, use_spb, use_dcse, 0.15)
        self.enc3 = ControlledSpatialBlock(256, 512, use_spb, use_dcse, 0.20)
        self.encoder_dropout = nn.Dropout2d(0.30)
        self.lat3 = nn.Conv2d(512, 256, 1)
        self.lat2 = nn.Conv2d(256, 256, 1)
        self.lat1 = nn.Conv2d(128, 256, 1)
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)
        self.dec1 = nn.Sequential(
            DSConv(320, 128), nn.Dropout2d(0.20), DSConv(128, 128)
        )
        self.dec2 = nn.Sequential(
            DSConv(192, 64), nn.Dropout2d(0.20), DSConv(64, 64)
        )
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x0_pool = F.max_pool2d(x0, 2)
        x1 = self.encoder_dropout(self.enc1(x0_pool))
        x2 = self.encoder_dropout(self.enc2(F.max_pool2d(x1, 2)))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        p2 = self.lat2(x2) + F.interpolate(
            self.lat3(x3),
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.lat1(x1) + F.interpolate(
            p2,
            size=x1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.smooth(p1)
        dec1 = self.dec1(
            torch.cat(
                (
                    F.interpolate(
                        fused,
                        size=x0_pool.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    x0_pool,
                ),
                dim=1,
            )
        )
        dec2 = self.dec2(
            torch.cat(
                (
                    F.interpolate(
                        dec1,
                        size=x0.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    x0,
                ),
                dim=1,
            )
        )
        return self.head(dec2)


def build_landslidenet_variant(
    name: str, num_bands: int, num_classes: int = 2
) -> LandslideNet:
    """Build LandslideNet or one of its three strictly controlled ablations."""
    variants = {
        "landslidenet": (True, True),
        "baseline": (False, False),
        "only_dcse": (False, True),
        "only_spb": (True, False),
    }
    try:
        use_spb, use_dcse = variants[name]
    except KeyError as error:
        raise ValueError(f"Not a LandslideNet variant: {name}") from error
    return LandslideNet(
        num_bands,
        num_classes,
        use_spb=use_spb,
        use_dcse=use_dcse,
    )


__all__ = [
    "ControlledSpatialBlock",
    "DCSELayer",
    "DSConv",
    "LandslideNet",
    "build_landslidenet_variant",
]
