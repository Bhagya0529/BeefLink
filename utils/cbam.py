"""
utils/cbam.py

Implements CBAM (Convolutional Block Attention Module) — a lightweight
attention mechanism that helps the network focus on the most informative
channels and spatial regions of a feature map. This is the module referenced
in the "YOLOv8-CBAM" cattle detection literature.

CBAM has two sequential parts:
    1. Channel Attention — "which channels (feature types) matter most?"
       Uses both average-pooled and max-pooled descriptors through a shared
       small MLP, so the network sees both "typical" and "most extreme"
       activation per channel.
    2. Spatial Attention  — "which spatial locations matter most?"
       Pools across channels (avg + max) and runs a 7x7 conv to produce a
       single-channel attention map over the image.

The output has the SAME shape as the input — CBAM only re-weights existing
features, it doesn't change channel count or spatial size. That's what lets
us drop it into a YOLO architecture yaml without disturbing the channel
bookkeeping the rest of the network relies on.

USAGE (registering with Ultralytics so the yaml parser can find "CBAM"):
    from utils.cbam import CBAM, register_cbam
    register_cbam()  # call this BEFORE loading any yaml that uses CBAM
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        reduced = max(channels // ratio, 8)  # avoid collapsing to 0 on small channel counts
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # shape (B, C, 1, 1) — one weight per channel


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(combined))          # (B, 1, H, W)


class CBAM(nn.Module):
    """
    Full CBAM block: channel attention applied first, then spatial attention
    on the re-weighted result. Output shape == input shape, so this can be
    inserted anywhere in a network without adjusting surrounding layers.
    """
    def __init__(self, c1: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def register_cbam():
    """
    Makes the CBAM class resolvable by name inside Ultralytics' YOLO yaml
    parser. Ultralytics builds models by reading a yaml like:

        [-1, 1, CBAM, [1024]]

    and looks up the string "CBAM" in the global namespace of its own
    ultralytics.nn.tasks module. Since CBAM is defined here, not inside
    Ultralytics' package, we inject it into that namespace at runtime.
    Call this once, before building/loading any model that uses a CBAM yaml.
    """
    import ultralytics.nn.tasks as tasks
    tasks.CBAM = CBAM
