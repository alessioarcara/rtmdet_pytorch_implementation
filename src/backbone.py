import torch.nn as nn
from torch import Tensor

from src.config import Config
from src.layers.conv_module import ConvModule


class CSPNext(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        widen_factor = cfg.widen_factor

        base = [32, 32, 64, 128]
        ch = [max(1, int(round(c * widen_factor))) for c in base]

        self.stem = nn.Sequential(
            ConvModule(in_channels=3, out_channels=ch[0], stride=2),
            ConvModule(in_channels=ch[0], out_channels=ch[1]),
            ConvModule(in_channels=ch[1], out_channels=ch[2]),
        )

        self.stage1 = nn.Sequential(
            ConvModule(in_channels=ch[2], out_channels=ch[3], stride=2)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        return x
