import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, generalized_box_iou_loss
from tqdm import tqdm

from rtmdet import RTMDet


class ShapesDataset(Dataset):
    """
    Draws either a square or circle at random position.
    0 -> square, 1 -> circle
    """

    def __init__(self, n: int, img_size: int, seed: int):
        self.n = n
        self.img_size = img_size
        self.rng = random.Random(seed)

    def _rand_box(self) -> Tuple[int, ...]:
        s = self.img_size
        rng = self.rng

        side = rng.randint(s // 8, s // 3)
        x1 = rng.randint(0, s - side - 1)
        y1 = rng.randint(0, s - side - 1)
        x2 = x1 + side
        y2 = y1 + side
        return x1, y1, x2, y2

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        label = 0 if self.rng.random() < 0.5 else 1
        x1, y1, x2, y2 = self._rand_box()

        if label == 0:
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        else:
            draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)

        target = {
            "bboxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.long),
        }

        return img_t, target


def collate_fn(batch: list):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


@dataclass
class TrainConfig:
    img_size: int = 128
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 4e-4
    weight_decay: float = 0.05


def best_iou_idx(
    pred_boxes: Tensor,  # [B, N, 4]
    gt_boxes: Tensor,  # [B, 4]
) -> Tensor:
    """
    finds, for each image in a batch, the predicted bounding box that has the
    highest IoU
    """
    B, N, _ = pred_boxes.shape
    out = []
    for b in range(B):
        ious = box_iou(pred_boxes[b], gt_boxes[b].view(1, 4)).squeeze(-1)
        out.append(torch.argmax(ious))
    return torch.stack(out).to(pred_boxes.device)


def main():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ShapesDataset(1000, cfg.img_size, 0)

    train_loader = DataLoader(train_ds, cfg.batch_size, True)

    model = RTMDet.from_preset("small", 2)
    model.to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for ep in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch", colour="green"):
        model.train()
        pbar = tqdm(
            train_loader, desc=f"Epoch {ep} Batches", leave=False, colour="blue"
        )
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gt_boxes = targets["bboxes"].to(device, non_blocking=True)  # [B, 4]
            gt_labels = targets["labels"].to(device, non_blocking=True)  # [B]

            optim.zero_grad(set_to_none=True)

            pred_boxes, pred_scores, _ = model(imgs)  # [B, N, 4], [B, 1, N], [B, N]

            B = pred_boxes.shape[0]

            # select best_proposals
            with torch.no_grad():
                best_idx = best_iou_idx(pred_boxes, gt_boxes)

            # box loss on best proposals
            pred_boxes_best = pred_boxes[torch.arange(B, device=device), best_idx]
            giou = generalized_box_iou_loss(
                gt_boxes, pred_boxes_best.unsqueeze(1), reduction="mean"
            )

            cls_outputs, _ = model._forward_raw(imgs)  # [B, C, H, W]

            logits_levels = []
            for cls_l in cls_outputs:
                # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
                logits_levels.append(
                    cls_l.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(cls_l.size(0), -1, cls_l.size(1))
                )
            pred_logits_all = torch.cat(logits_levels, dim=1)  # [B, N, C]

            pred_logits_best = pred_logits_all[
                torch.arange(B, device=device), best_idx
            ]  # [B, C]

            ce = nn.CrossEntropyLoss()(
                pred_logits_best, gt_labels.squeeze(-1).to(torch.long)
            )

            loss = 2.0 * giou + 1.0 * ce

            loss.backward()
            optim.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                giou=f"{giou.item():.4f}",
                ce=f"{ce.item():.4f}",
            )


if __name__ == "__main__":
    main()
