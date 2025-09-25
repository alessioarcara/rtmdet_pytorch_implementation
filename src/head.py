from typing import Tuple

import torch.nn as nn
from torch import Tensor

from src.config import Config
from src.layers import ConvModule
from src.utils import apply_factor


class RTMDetHead(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.cfg = cfg
        c = apply_factor(cfg.neck_out_channels, cfg.widen_factor)

        # Per-level towers
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        # Per-level prediction heads
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        for _ in range(cfg.head_num_levels):
            cls_tower = nn.ModuleList()
            reg_tower = nn.ModuleList()

            for _ in range(cfg.head_num_stacked_convs):
                cls_tower.append(
                    ConvModule(
                        c_in=c,
                        c_out=c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                reg_tower.append(
                    ConvModule(
                        c_in=c,
                        c_out=c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            self.cls_convs.append(cls_tower)
            self.reg_convs.append(reg_tower)

            self.rtm_cls.append(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    in_channels=c, out_channels=4, kernel_size=1, stride=1, padding=0
                )
            )

        for i in range(cfg.head_num_levels):
            for j in range(cfg.head_num_stacked_convs):
                self.cls_convs[i][j].conv = self.cls_convs[0][j].conv
                self.reg_convs[i][j].conv = self.reg_convs[0][j].conv

    def forward(self, x: Tuple[Tensor, ...]) -> tuple:
        """ """
        cls_scores, bbox_preds = [], []

        for i, feat in enumerate(x):
            # ---- classification path ----
            cls_feat = feat
            for layer in self.cls_convs[i]:
                cls_feat = layer(cls_feat)
            cls_scores.append(self.rtm_cls[i](cls_feat))

            # ---- regression path ----
            reg_feat = feat
            for layer in self.reg_convs[i]:
                reg_feat = layer(reg_feat)
            bbox_preds.append(self.rtm_reg[i](reg_feat))

        return tuple(cls_scores), tuple(bbox_preds)
