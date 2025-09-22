from typing import Optional

import torch.nn as nn
from torch import Tensor


class ConvModule(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DWConvModule(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        self.depthwise_conv = ConvModule(
            c_in=c_in,
            c_out=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=c_out,
        )
        self.pointwise_conv = ConvModule(
            c_in=c_out, c_out=c_out, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
