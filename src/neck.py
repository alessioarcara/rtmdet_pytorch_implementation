from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.config import Config
from src.layers import ConvModule, CSPLayer


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        deepen = cfg.deepen_factor
        widen = cfg.widen_factor

        base_channels = [256, 512, 1024]
        ch = [max(1, int(round(c * widen))) for c in base_channels]
        self.ch = ch

        # top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for i in range(len(base_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    c_in=ch[i],
                    c_out=ch[i - 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    c_in=ch[i - 1] * 2,
                    c_out=ch[i - 1],
                    n=max(1, int(round(3 * deepen))),
                    add=False,
                    use_attention=False,
                )
            )

        # bottom-up blocks

        # out convs

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # top-down path
        inner_outs = [inputs[-1]]
        num_levels = len(self.ch)
        for i in range(num_levels - 1, 0, -1):
            high = inner_outs[0]  # current top feature (deep)
            low = inputs[i - 1]  # skip connection from lower level

            # 1x1 conv to match channel dimensions
            high_reduced = self.reduce_layers[num_levels - 1 - i](high)
            # Upsample to the resolution of low
            high_up = self.upsample(high_reduced)

            # Fuse and refine with CSP block
            fused = torch.cat([high_up, low], dim=1)
            out = self.top_down_blocks[num_levels - 1 - i](fused)

            inner_outs.insert(0, out)

        # bottom-up path
        return None
