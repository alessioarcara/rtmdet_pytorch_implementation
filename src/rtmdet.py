import torch.nn as nn
from torch import Tensor

from src.backbone import CSPNext
from src.config import Config
from src.head import RTMDetHead
from src.neck import CSPNeXtPAFPN


class RTMDet(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.backbone = CSPNext(cfg=cfg)
        self.neck = CSPNeXtPAFPN(cfg=cfg)
        self.bbox_head = RTMDetHead(cfg=cfg, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head(x)
        return x
