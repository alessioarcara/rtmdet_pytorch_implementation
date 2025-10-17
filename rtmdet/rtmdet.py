from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import PositiveInt, validate_call
from torch import Tensor
from torchvision.ops import batched_nms

from rtmdet.backbone import CSPNext
from rtmdet.checkpoint_utils import load_and_verify_weight, load_mmdet_checkpoint
from rtmdet.config import RTMDetConfig
from rtmdet.constants import _PRETRAINED_URLS
from rtmdet.head import RTMDetHead
from rtmdet.neck import CSPNeXtPAFPN
from rtmdet.typings import PresetName


class RTMDet(nn.Module):
    def __init__(self, cfg: RTMDetConfig, separate_outputs: bool = True):
        super().__init__()
        self.backbone = CSPNext(cfg=cfg)
        self.neck = CSPNeXtPAFPN(cfg=cfg)
        self.bbox_head = RTMDetHead(cfg=cfg)

        self.stage = [80, 40, 20]
        self.export_mode = False

        self.max_num_detections = cfg.max_num_detections
        self.separate_outputs = separate_outputs

    def _forward_raw(self, x: Tensor) -> Tuple[List[Tensor], ...]:
        x = self.backbone(x)
        x = self.neck(x)
        cls_outputs, box_outputs = self.bbox_head(x)
        return cls_outputs, box_outputs

    def forward(
        self, x: Tensor
    ) -> Union[
        Tensor,
        Tuple[List[Tensor], ...],  # training
        Tuple[Tensor, ...],
    ]:
        if self.export_mode:
            return self.predict(x)
        return self._forward_raw(x)

    def predict(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Applies postprocessing and NMS.

        Returns:
          - se separate_outputs=False: Tensor [B, N, 6] con [x1, y1, x2, y2, conf, class]
          - se separate_outputs=True:  (boxes_scores [B, N, 5], class_idx [B, N])
        """
        cls_outputs, box_outputs = self._forward_raw(x)
        device = x.device
        # Restituisce due tensori:
        # - Il primo ha un numero di canali pari al numero di classi.
        # - Il secondo ha 4 canali che rappresentano le coordinate della bounding box nel formato (x1, y1, x2, y2).

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
            # Ogni box: [x1_off, y1_off, x2_off, y2_off, conf, class]
            box = torch.cat([box, conf, class_idx], dim=-1)  # [B, H, W, 6]

            # Calcola dimensione di una cella in pixel
            step = self.input_shape // self.stage[i]

            # Crea un vettore di coordinate delle celle nella griglia
            grid = torch.arange(self.stage[i], device=device) * step
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

        boxes, scores, classes = self.batch_nms(result_box)
        boxes_with_scores = torch.cat((boxes, scores), dim=-1)  # [B, K, 5]

        return boxes_with_scores, classes

    def batch_nms(
        self,
        result_box: Tensor,  # [B, N, 6]
    ) -> Tuple[Tensor, ...]:
        B = result_box.size(0)

        boxes = result_box[:, :, :4]  # [x1,y1,x2,y2]
        scores = result_box[:, :, 4]  # [conf]
        classes = result_box[:, :, 5].to(torch.long)  # [class]

        batch_boxes, batch_scores, batch_classes = [], [], []

        for i in range(B):
            keep = batched_nms(
                boxes[i],
                scores[i],
                classes[i],
                iou_threshold=0.5,
            )[: self.max_num_detections]

            batch_boxes.append(boxes[i][keep])  # [max_num, 4]
            batch_scores.append(scores[i][keep].unsqueeze(-1))  # [max_num, 1]
            batch_classes.append(classes[i][keep])  # [max_num]

        final_boxes = torch.stack(batch_boxes, dim=0)  # [B, max_num, 4]
        final_scores = torch.stack(batch_scores, dim=0)  # [B, max_num, 1]
        final_classes = torch.stack(batch_scores, dim=0)  # [B, max_num]
        return final_boxes, final_scores, final_classes

    @classmethod
    @validate_call
    def from_preset(
        cls,
        name: PresetName,
        num_classes: Optional[PositiveInt] = None,
        pretrained: bool = True,
    ) -> Any:
        cfg = RTMDetConfig.from_preset(name)
        if num_classes is not None:
            cfg.num_classes = num_classes

        model = cls(cfg)

        if pretrained:
            url = _PRETRAINED_URLS[name]
            cache_dir = Path(torch.hub.get_dir()) / "rtmdet"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = cache_dir / Path(url).name

        if not ckpt_path.exists():
            torch.hub.download_url_to_file(url, ckpt_path)

        state_dict = load_mmdet_checkpoint(ckpt_path)
        load_and_verify_weight(model, state_dict)

        return model
