"""LandslideNet architecture defined by Fig. 5 of the revised manuscript.

The encoder is
``CBR--MaxPool--SPB*--MaxPool--SPB*--MaxPool--SPB--MaxPool``.  The three
pre-pooling SPB outputs and the final pooled feature are fused by an FPN.
An asterisk denotes dropout in the manuscript figure.
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


class CBR(nn.Sequential):
    """The convolution--BatchNorm--ReLU stem in Fig. 5(a)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DCSELayer(nn.Module):
    """Dynamic channel squeeze-and-excitation module in Fig. 5(d)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        # The learnable channel-reorganisation matrix is softmax-normalised
        # before multiplication, as explicitly shown in Fig. 5(d).
        self.phi = nn.Parameter(torch.empty(hidden, channels))
        nn.init.kaiming_uniform_(self.phi, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.theta(x).flatten(1)
        channel_weights = torch.matmul(theta, F.softmax(self.phi, dim=-1))
        return x * channel_weights.sigmoid()[:, :, None, None]


class SpatialPerceptionBlock(nn.Module):
    """Deformable convolution--GroupNorm--DCSE residual SPB."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_spb: bool = True,
        use_dcse: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_spb = bool(use_spb)
        if self.use_spb:
            # A 3x3 deformable kernel needs x/y offsets for nine positions:
            # 2 * 3 * 3 = 18 channels. Zero initialization starts as an
            # ordinary convolution and is preserved by the global initializer.
            self.offset = nn.Conv2d(in_channels, 18, 3, padding=1)
            self.offset._landslidenet_zero_init = True
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.spatial = DeformConv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            )
        else:
            self.offset = None
            self.spatial = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            )
        self.norm = nn.GroupNorm(_valid_groups(out_channels), out_channels)
        self.channel = DCSELayer(out_channels) if use_dcse else nn.Identity()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset is None:
            out = self.spatial(x)
        else:
            out = self.spatial(x, self.offset(x))
        out = self.channel(self.norm(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return self.dropout(out)


class LandslideNet(nn.Module):
    """Four-downsampling LandslideNet with SPB, DCSE, and FPN."""

    def __init__(
        self,
        num_bands: int,
        num_classes: int = 2,
        *,
        use_spb: bool = True,
        use_dcse: bool = True,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.stem = CBR(num_bands, 64)
        self.spb1 = SpatialPerceptionBlock(
            64,
            128,
            use_spb=use_spb,
            use_dcse=use_dcse,
            dropout=dropout,
        )
        self.spb2 = SpatialPerceptionBlock(
            128,
            256,
            use_spb=use_spb,
            use_dcse=use_dcse,
            dropout=dropout,
        )
        self.spb3 = SpatialPerceptionBlock(
            256,
            512,
            use_spb=use_spb,
            use_dcse=use_dcse,
        )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        # FPN inputs correspond exactly to the three arrows from the SPBs plus
        # the deepest MaxPool output in Fig. 5(a).
        self.lat1 = nn.Conv2d(128, 256, 1)
        self.lat2 = nn.Conv2d(256, 256, 1)
        self.lat3 = nn.Conv2d(512, 256, 1)
        self.lat4 = nn.Conv2d(512, 256, 1)
        self.fpn_smooth = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(256, num_classes, 1)
        self.head._landslidenet_classification_head = True

    @staticmethod
    def _upsample(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        stem = self.stem(x)
        feature1 = self.spb1(self.pool1(stem))
        feature2 = self.spb2(self.pool2(feature1))
        feature3 = self.spb3(self.pool3(feature2))
        feature4 = self.pool4(feature3)

        pyramid4 = self.lat4(feature4)
        pyramid3 = self.lat3(feature3) + self._upsample(pyramid4, feature3)
        pyramid2 = self.lat2(feature2) + self._upsample(pyramid3, feature2)
        pyramid1 = self.lat1(feature1) + self._upsample(pyramid2, feature1)
        fused = self.fpn_dropout(self.fpn_smooth(pyramid1))
        fused = F.interpolate(
            fused,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.head(fused)


def build_landslidenet_variant(
    name: str,
    num_bands: int,
    num_classes: int = 2,
) -> LandslideNet:
    """Build the proposed model or a controlled manuscript ablation."""
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
    "CBR",
    "DCSELayer",
    "SpatialPerceptionBlock",
    "LandslideNet",
    "build_landslidenet_variant",
]
