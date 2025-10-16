from typing import Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.backbone import CSPNext
from src.config import Config
from src.head import RTMDetHead
from src.neck import CSPNeXtPAFPN


class RTMDet(nn.Module):
    def __init__(self, cfg: Config, num_classes: int, separate_outputs: bool = True):
        super().__init__()
        self.backbone = CSPNext(cfg=cfg)
        self.neck = CSPNeXtPAFPN(cfg=cfg)
        self.bbox_head = RTMDetHead(cfg=cfg, num_classes=num_classes)

        self.input_shape = 640
        self.stage = [80, 40, 20]
        self.separate_outputs = separate_outputs

    def _forward(self, x: Tensor) -> Tuple[List[Tensor], ...]:
        """
        Output:
        - se separate_outputs=False: Tensor [B, N, 6] con [x1, y1, x2, y2, conf, class]
        - se separate_outputs=True:  (boxes_scores [B, N, 5], class_idx [B, N])
        """
        x = self.backbone(x)
        x = self.neck(x)
        cls_outputs, box_outputs = self.bbox_head.forward(x)
        return cls_outputs, box_outputs

    def predict(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Output:
          - se separate_outputs=False: Tensor [B, N, 6] con [x1, y1, x2, y2, conf, class]
          - se separate_outputs=True:  (boxes_scores [B, N, 5], class_idx [B, N])
        """
        cls_outputs, box_outputs = self._forward(x)

        boxes_list = []
        B = x.shape[0]

        # Itera su ogni stage della feature pyramid
        for i, (cls, box) in enumerate(zip(cls_outputs, box_outputs)):
            # [B, C, H, W] -> [B, H, W, C]
            cls = cls.permute(0, 2, 3, 1).contiguous()
            box = box.permute(0, 2, 3, 1).contiguous()

            # Probabilita' in [0,1]
            cls = torch.sigmoid(cls)

            # conf = max su classi; class_idx = indice classe max
            conf, class_idx = torch.max(cls, dim=3, keepdim=True)
            class_idx = class_idx.to(torch.float32)

            # Unisce box offsets, class index e confidence
            # Ogni box: [x1_off, y1_off, x2_off, y2_off, class, conf]
            box = torch.cat([box, class_idx, conf], dim=-1)  # [B, H, W, 6]

            # Calcola dimensione di una cella in pixel
            step = self.input_shape // self.stage[i]

            # Crea un vettore di coordinate delle celle nella griglia
            grid = torch.arange(self.stage[i]) * step
            # Crea coordinate x e y della griglia
            # gx =
            # [[0, 32, 64],
            # [0, 32, 64],
            # [0, 32, 64]]
            # gy =
            # [[0, 0, 0],
            # [32, 32, 32],
            # [64, 64, 64]]
            gx, gy = torch.meshgrid(grid, grid, indexing="xy")
            # block contiene le coordinate (x, y) del punto di riferimento in pixel della cella (y, x) nella griglia
            # block =
            # [
            #  [[ [0, 0], [32, 0], [64, 0] ],
            #   [ [0,32], [32,32], [64,32] ],
            #   [ [0,64], [32,64], [64,64] ]]
            # ]
            block = torch.stack([gx, gy], dim=-1)

            # Aggiusta le coordinate delle box rispetto alla griglia
            box[..., :2] = block - box[..., :2]  # top-left
            box[..., 2:4] = block + box[..., 2:4]  # bottom-right

            # [B, H*W, 6]
            box = box.reshape(B, -1, 6)
            boxes_list.append(box)

        result_box = torch.cat(boxes_list, dim=1)

        if not self.separate_outputs:
            return result_box

        boxes_scores: torch.Tensor = result_box[:, :, :5]  # [x1,y1,x2,y2,conf]
        class_idx = result_box[:, :, 5].to(torch.long)  # [class]

        return boxes_scores, class_idx
