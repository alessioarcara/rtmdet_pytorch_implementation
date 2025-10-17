import torch.nn as nn
from torch import Tensor

from rtmdet.layers.conv_module import ConvModule, DWConvModule


class CSPNextBlock(nn.Module):
    """CSPNeXt block with 5x5 depthwise conv for larger receptive field"""

    def __init__(self, c: int, add: bool):
        super().__init__()
        self.add = add

        self.conv1 = ConvModule(c_in=c, c_out=c, kernel_size=3, stride=1, padding=1)
        self.conv2 = DWConvModule(c_in=c, c_out=c, kernel_size=5, stride=1, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)

        if self.add:
            out = out + residual
        return out
