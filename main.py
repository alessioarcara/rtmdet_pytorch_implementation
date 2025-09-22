import torch

from src.backbone import CSPNext
from src.checkpoint_utils import (
    extract_sub_state_dict,
    load_mmdet_checkpoint,
    print_state_dict,
    check_params_updated,
)
from src.config import load_config


def main():
    sd = load_mmdet_checkpoint("./checkpoints/rtmdet_tiny.pth")
    sd_backbone = extract_sub_state_dict(sd, "backbone.")
    print_state_dict(sd_backbone)

    cfg = load_config("./configs/rtmdet_tiny.yaml")
    backbone = CSPNext(cfg=cfg)

    check_params_updated(model=backbone, sd=sd_backbone)

    x = torch.randn(1, 3, 640, 640)
    c3, c4, c5 = backbone(x)


if __name__ == "__main__":
    main()
