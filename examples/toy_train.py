from rtmdet import RTMDet
import torch
from torch.utils.data.dataset import Dataset
from typing import Tuple
from PIL import Image, ImageDraw
import random
import numpy as np


class ShapesDetDataset(Dataset):
    """
    Draws either a square or circle at random position.
    0 -> square, 1 -> circle
    """

    def __init__(self, n: int = 1000, img_size: int = 640, seed: int = 0):
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

    def __getitem__(self, _: int) -> torch.Tensor:
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


def main():
    model = RTMDet.from_preset("tiny", 30)

    x = torch.randn(24, 3, 640, 640)
    # out = model.predict(x)
    # print(out)


if __name__ == "__main__":
    main()
