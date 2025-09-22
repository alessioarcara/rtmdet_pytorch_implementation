import torch
import torch.nn as nn
from torch import Tensor

from src.layers.channel_attention import ChannelAttention
from src.layers.conv_module import ConvModule
from src.layers.csp_next_block import CSPNextBlock


class CSPLayer(nn.Module):
    def __init__(self, add: bool, c: int, n: int):
        super().__init__()

        c_half = c // 2

        self.main_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c, c_out=c_half
        )
        self.short_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c, c_out=c_half
        )
        self.final_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c, c_out=c
        )

        self.blocks = nn.Sequential()
        for _ in range(n):
            self.blocks.append(CSPNextBlock(c=c_half, add=add))

        self.attention = ChannelAttention(c=c)

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x = torch.cat([x_main, x_short], dim=1)
        x = self.attention(x)
        x = self.final_conv(x)
        return x
