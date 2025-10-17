from typing import Tuple

import torch.nn as nn
from torch import Tensor

from rtmdet.config import RTMDetConfig
from rtmdet.layers import ConvModule, CSPLayer, SPFFBottleneck
from rtmdet.utils import apply_factor


class CSPNext(nn.Module):
    def __init__(self, cfg: RTMDetConfig):
        super().__init__()

        ch = apply_factor([32, 32, 64, 128, 256, 512, 1024], cfg.widen_factor)
        depths = apply_factor([3, 6, 6, 3], cfg.deepen_factor)

        self.stem = nn.Sequential(
            ConvModule(c_in=3, c_out=ch[0], stride=2),
            ConvModule(c_in=ch[0], c_out=ch[1]),
            ConvModule(c_in=ch[1], c_out=ch[2]),
        )

        self.stage1 = nn.Sequential(
            ConvModule(c_in=ch[2], c_out=ch[3], stride=2),
            CSPLayer(c_in=ch[3], c_out=ch[3], n=depths[0], add=True),
        )

        self.stage2 = nn.Sequential(
            ConvModule(c_in=ch[3], c_out=ch[4], stride=2),
            CSPLayer(c_in=ch[4], c_out=ch[4], n=depths[1], add=True),
        )

        self.stage3 = nn.Sequential(
            ConvModule(c_in=ch[4], c_out=ch[5], stride=2),
            CSPLayer(c_in=ch[5], c_out=ch[5], n=depths[2], add=True),
        )

        self.stage4 = nn.Sequential(
            ConvModule(c_in=ch[5], c_out=ch[6], stride=2),
            SPFFBottleneck(c=ch[6]),
            CSPLayer(c_in=ch[6], c_out=ch[6], n=depths[3], add=False),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.stem(x)
        x = self.stage1(x)
        stride8 = self.stage2(x)
        stride16 = self.stage3(stride8)
        stride32 = self.stage4(stride16)
        return stride8, stride16, stride32
