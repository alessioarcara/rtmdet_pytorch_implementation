import torch
import torch.nn as nn
from torch import Tensor

from rtmdet.layers.conv_module import ConvModule


class SPFFBottleneck(nn.Module):
    """
    SPFF (Spatial Pyramid Pooling - Fast) bottleneck module.

    Lightweight alternative to classic Spatial Pyramid Pooling (SPP). Instead of
    using multiple large pooling kernels in parallel, SPFF applies a smaller
    max-pooling kernel (e.g., 5x5) several times sequentially, which effectively
    enlarges the receptive field with reduced computational cost.
    """

    def __init__(self, c: int):
        super().__init__()

        c_half = c // 2

        self.conv1 = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c, c_out=c_half
        )
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c_half * 4, c_out=c
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        x1 = self.pool(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.conv2(out)
        return out
