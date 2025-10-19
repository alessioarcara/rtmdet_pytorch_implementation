from pathlib import Path
from typing import Any, List, Optional, Tuple

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
    def __init__(self, cfg: RTMDetConfig):
        super().__init__()
        self.backbone = CSPNext(cfg=cfg)
        self.neck = CSPNeXtPAFPN(cfg=cfg)
        self.bbox_head = RTMDetHead(cfg=cfg)

        self.export_mode = False
        self.input_shape = cfg.input_size
        self.score_threshold = cfg.score_threshold
        self.nms_iou_threshold = cfg.nms_iou_threshold
        self.max_num_detections = cfg.max_num_detections

    def _forward_raw(self, x: Tensor) -> Tuple[List[Tensor], ...]:
        """
        Returns raw head outputs:
            - cls_outputs : per-level class logits [B, C_cls, H_l, W_l]
            - box_outputs : per-level box offsets [B, 4, H_l, W_l]
        """
        x = self.backbone(x)
        x = self.neck(x)
        cls_outputs, box_outputs = self.bbox_head(x)
        return cls_outputs, box_outputs

    def forward(self, x: Tensor, return_logits: bool = False) -> Tuple[Tensor, ...]:
        """
        Returns:
          - export_mode=False, return_logits=False:
              boxes[B,N,4], scores[B,N,1], classes[B,N]
          - export_mode=False, return_logits=True:
              boxes, scores, classes, logits[B,N,C]
          - export_mode=True:
              boxes_with_scores[B,max_num,5], classes[B,max_num]
        """
        cls_outputs, box_outputs = self._forward_raw(x)

        boxes, scores, classes = self.decode(cls_outputs, box_outputs)

        if self.export_mode:
            boxes, scores, classes = self.nms(boxes, scores, classes)
            boxes_with_scores = torch.cat((boxes, scores), dim=-1)  # [B, max_num, 5]
            return boxes_with_scores, classes

        if return_logits:
            flat_logits = []
            for lvl in cls_outputs:  # [B, C, H, W]
                B, C, H, W = lvl.shape
                flat = lvl.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
                flat_logits.append(flat)
            logits = torch.cat(flat_logits, dim=1)  # [B, N, C]
            return boxes, scores, classes, logits

        return boxes, scores, classes

    def decode(
        self, cls_outputs: List[Tensor], box_outputs: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """
        Converts per-pyramid-level raw head outputs into
        absolute [x1, y1, x2, y2], score, class tensors
        """
        assert len(cls_outputs) == len(box_outputs), "cls/box levels mismatch"
        device = cls_outputs[0].device
        B = cls_outputs[0].shape[0]

        boxes_list = []
        for i, (cls, box) in enumerate(zip(cls_outputs, box_outputs)):
            # [B, C, H, W] -> [B, H, W, C]
            cls = cls.permute(0, 2, 3, 1).contiguous()
            box = box.permute(0, 2, 3, 1).contiguous()

            # logits -> probabilities
            cls = torch.sigmoid(cls)

            # per-location best class + score
            conf, class_idx = torch.max(cls, dim=3, keepdim=True)
            class_idx = class_idx.to(torch.float32)

            # Combine box offsets, confidence, and class index
            # Each box: [x1_off, y1_off, x2_off, y2_off, conf, class]
            box = torch.cat([box, conf, class_idx], dim=-1)  # [B, H, W, 6]

            # Compute grid step size in pixels
            stage = box.shape[1]
            step = self.input_shape // stage

            # Create grid coordinates
            grid = torch.arange(stage, device=device) * step
            # gx =
            # [[0, 32, 64],
            # [0, 32, 64],
            # [0, 32, 64]]
            # gy =
            # [[0, 0, 0],
            # [32, 32, 32],
            # [64, 64, 64]]
            gx, gy = torch.meshgrid(grid, grid, indexing="xy")
            # block (y, x) contains the reference point (x, y) in pixel space
            # block =
            # [
            #  [[ [0, 0], [32, 0], [64, 0] ],
            #   [ [0,32], [32,32], [64,32] ],
            #   [ [0,64], [32,64], [64,64] ]]
            # ]
            block = torch.stack([gx, gy], dim=-1)

            # Adjust predicted offsets relative to grid position
            box[..., :2] = block - box[..., :2]  # top-left
            box[..., 2:4] = block + box[..., 2:4]  # bottom-right

            # Flatten spatial dimensions
            # [B, H*W, 6]
            box = box.reshape(B, -1, 6)
            boxes_list.append(box)

        # Concatenate all levels
        result_box = torch.cat(boxes_list, dim=1)

        boxes = result_box[..., :4]  # [x1,y1,x2,y2]
        scores = result_box[..., 4:5]  # [conf]
        classes = result_box[..., 5].to(torch.long)  # [class]

        # Clamp box coordinates to image bounds to prevent negatives or out-of-range values
        boxes[..., 0::2] = boxes[..., 0::2].clamp_(0, self.input_shape - 1)
        boxes[..., 1::2] = boxes[..., 1::2].clamp_(0, self.input_shape - 1)

        return boxes, scores, classes

    def nms(
        self,
        boxes: Tensor,  # [B, N, 4]
        scores: Tensor,  # [B, N]
        classes: Tensor,  # [B, N]
    ) -> Tuple[Tensor, ...]:
        """
        Class-wise nms per image
        """
        B = boxes.shape[0]

        batch_boxes, batch_scores, batch_classes = [], [], []
        for i in range(B):
            im_boxes = boxes[i]
            im_scores = scores[i]
            im_classes = classes[i]

            # threshold before nms to reduce work
            keep = im_scores >= float(self.score_threshold)
            im_boxes = im_boxes[keep]
            im_scores = im_scores[keep]
            im_classes = im_classes[keep]

            keep = batched_nms(
                boxes[i],
                scores[i],
                classes[i],
                iou_threshold=self.nms_iou_threshold,
            )[: self.max_num_detections]

            batch_boxes.append(boxes[i][keep])  # [max_num, 4]
            batch_scores.append(scores[i][keep].unsqueeze(-1))  # [max_num, 1]
            batch_classes.append(classes[i][keep])  # [max_num]

        return (
            torch.stack(batch_boxes, 0),
            torch.stack(batch_scores, 0),
            torch.stack(batch_classes, 0),
        )

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
                torch.hub.download_url_to_file(url, str(ckpt_path))

            state_dict = load_mmdet_checkpoint(str(ckpt_path))
            load_and_verify_weight(model, state_dict)

        return model
