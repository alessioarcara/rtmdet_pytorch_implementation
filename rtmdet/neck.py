from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rtmdet.config import RTMDetConfig
from rtmdet.layers import ConvModule, CSPLayer
from rtmdet.utils import apply_factor


class CSPNeXtPAFPN(nn.Module):
    """
    PAFPN-style neck with CSP blocks (similar to YOLO)
    """

    def __init__(self, cfg: RTMDetConfig):
        super().__init__()

        ch = apply_factor([256, 512, 1024], cfg.widen_factor)
        depth = apply_factor(3, cfg.deepen_factor)
        neck_out_channels = apply_factor(cfg.neck_out_channels, cfg.widen_factor)

        self.num_levels = len(ch)

        # ---- Top-down path ----
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()  # 1x1 conv to reduce channels of "high_up"
        self.top_down_blocks = (
            nn.ModuleList()
        )  # CSP block on cat([high_up, low], dim=1)

        for i in range(self.num_levels - 1, 0, -1):
            # Reduce C_{i} -> C_{i-1}
            self.reduce_layers.append(
                ConvModule(
                    c_in=ch[i],
                    c_out=ch[i - 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # CSP on (C_{i-1} * 2) -> {C_{i-1}}
            self.top_down_blocks.append(
                CSPLayer(
                    c_in=ch[i - 1] * 2,
                    c_out=ch[i - 1],
                    n=depth,
                    add=False,
                    use_attention=False,
                )
            )

        # ---- Bottom-up path ----
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.downsamples.append(
                ConvModule(c_in=ch[i], c_out=ch[i], kernel_size=3, stride=2, padding=1)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    c_in=ch[i] * 2,
                    c_out=ch[i + 1],
                    n=depth,
                    add=False,
                    use_attention=False,
                )
            )

        # ---- Output convs ----
        self.out_convs = nn.ModuleList()

        for i in range(self.num_levels):
            self.out_convs.append(
                ConvModule(
                    c_in=ch[i],
                    c_out=neck_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # --- Top-down path ---
        inner_outs = [inputs[-1]]
        for i in range(self.num_levels - 1, 0, -1):
            high = inner_outs[0]  # current top feature (deep)
            low = inputs[i - 1]  # skip connection from lower level

            idx = (self.num_levels - 1) - i
            # 1x1 conv to match channel dimensions
            high_reduced = self.reduce_layers[idx](high)
            inner_outs[0] = high_reduced

            # Upsample to the resolution of low
            high_up = self.upsample(high_reduced)

            # Fuse and refine with CSP block
            fused = torch.cat([high_up, low], dim=1)
            inner_out = self.top_down_blocks[idx](fused)

            inner_outs.insert(0, inner_out)

        # Now inner_outs matches the order of inputs (high -> low resolution)

        # --- Bottom-up path ---
        outs = [inner_outs[0]]  # start from the highest-resolution refined feature
        for i in range(self.num_levels - 1):
            low = outs[-1]
            high = inner_outs[i + 1]
            low_down = self.downsamples[i](low)
            fused = torch.cat([low_down, high], dim=1)
            out = self.bottom_up_blocks[i](fused)
            outs.append(out)

        # --- Output convs ---
        for i in range(self.num_levels):
            outs[i] = self.out_convs[i](outs[i])

        return tuple(outs)
