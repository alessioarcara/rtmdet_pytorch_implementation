import torch
import torch.nn as nn
from torch import Tensor

from rtmdet.layers.channel_attention import ChannelAttention
from rtmdet.layers.conv_module import ConvModule
from rtmdet.layers.csp_next_block import CSPNextBlock


class CSPLayer(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, n: int, add: bool, use_attention: bool = False
    ):
        super().__init__()
        assert c_out % 2 == 0, "CSPLayer richiede c_out pari"

        c_half = c_out // 2

        self.main_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c_in, c_out=c_half
        )
        self.short_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c_in, c_out=c_half
        )
        self.final_conv = ConvModule(
            kernel_size=1, stride=1, padding=0, c_in=c_out, c_out=c_out
        )

        blocks = [CSPNextBlock(c=c_half, add=add) for _ in range(n)]
        self.blocks = nn.Sequential(*blocks)

        self.attention = ChannelAttention(c=c_out) if use_attention else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x = torch.cat([x_main, x_short], dim=1)
        return self.final_conv(self.attention(x))
