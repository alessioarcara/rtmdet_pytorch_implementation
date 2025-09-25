import torch

from src.rtmdet import RTMDet
from src.checkpoint_utils import (
    extract_sub_state_dict,
    load_mmdet_checkpoint,
    print_state_dict,
    check_params_updated,
)
from src.config import load_config


def main():
    sd = load_mmdet_checkpoint("./checkpoints/rtmdet_tiny.pth")
    cfg = load_config("./configs/rtmdet_tiny.yaml")
    model = RTMDet(cfg=cfg, num_classes=80)

    check_params_updated(model=model, sd=sd)

    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    print(out)


if __name__ == "__main__":
    main()
