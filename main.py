import torch

from src.backbone import CSPNext
from src.neck import CSPNeXtPAFPN
from src.checkpoint_utils import (
    extract_sub_state_dict,
    load_mmdet_checkpoint,
    print_state_dict,
    check_params_updated,
)
from src.config import load_config


def main():
    sd = load_mmdet_checkpoint("./checkpoints/rtmdet_tiny.pth")
    print_state_dict(sd)
    sd_backbone = extract_sub_state_dict(sd, "backbone.")
    sd_neck = extract_sub_state_dict(sd, "neck.")

    cfg = load_config("./configs/rtmdet_tiny.yaml")
    backbone = CSPNext(cfg=cfg)
    neck = CSPNeXtPAFPN(cfg=cfg)

    check_params_updated(model=backbone, sd=sd_backbone)
    check_params_updated(model=neck, sd=sd_neck)

    x = torch.randn(1, 3, 640, 640)
    stride8, stride16, stride32 = backbone(x)


if __name__ == "__main__":
    main()
