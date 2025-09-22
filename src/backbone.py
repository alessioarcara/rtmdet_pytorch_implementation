import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from src.config import Config
from src.layers import ConvModule, CSPLayer, SPFFBottleneck


class CSPNext(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        deepen = cfg.deepen_factor
        widen = cfg.widen_factor

        base_channels = [32, 32, 64, 128, 256, 512, 1024]
        ch = [max(1, int(round(c * widen))) for c in base_channels]

        base_depths = [3, 6, 6, 3]
        depths = [max(1, int(round(d * deepen))) for d in base_depths]

        self.stem = nn.Sequential(
            ConvModule(c_in=3, c_out=ch[0], stride=2),
            ConvModule(c_in=ch[0], c_out=ch[1]),
            ConvModule(c_in=ch[1], c_out=ch[2]),
        )

        self.stage1 = nn.Sequential(
            ConvModule(c_in=ch[2], c_out=ch[3], stride=2),
            CSPLayer(add=True, c=ch[3], n=depths[0]),
        )

        self.stage2 = nn.Sequential(
            ConvModule(c_in=ch[3], c_out=ch[4], stride=2),
            CSPLayer(add=True, c=ch[4], n=depths[1]),
        )

        self.stage3 = nn.Sequential(
            ConvModule(c_in=ch[4], c_out=ch[5], stride=2),
            CSPLayer(add=True, c=ch[5], n=depths[2]),
        )

        self.stage4 = nn.Sequential(
            ConvModule(c_in=ch[5], c_out=ch[6], stride=2),
            SPFFBottleneck(c=ch[6]),
            CSPLayer(add=False, c=ch[6], n=depths[3]),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.stem(x)
        x = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x2, x3, x4


if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)
    model = CSPNext(Config(deepen_factor=0.167, widen_factor=0.375))
    out1, out2, out3 = model(x)
    print(out1.shape, out2.shape, out3.shape)
