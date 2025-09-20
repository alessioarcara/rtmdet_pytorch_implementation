import torch
from torch.serialization import add_safe_globals
from numpy import dtype as np_dtype
from numpy.core.multiarray import _reconstruct as np_reconstruct, scalar as np_scalar  # type: ignore
import numpy as np
from src.utils import extract_sub_state_dict, print_state_dict
from src.layers.conv_module import ConvModule
from src.backbone import CSPNext
from src.config import load_config


HistoryBufferDummy = type("HistoryBuffer", (), {})
HistoryBufferDummy.__module__ = "mmengine.logging.history_buffer"


def main():
    add_safe_globals(
        [
            HistoryBufferDummy,
            np_dtype,
            np_scalar,
            np_reconstruct,
            np.ndarray,
            np.float64,
            np.dtypes.Float64DType,
            np.dtypes.Int64DType,
        ]
    )

    ckpt = torch.load(
        "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
        map_location="cpu",
        weights_only=True,
    )

    sd = ckpt.get("state_dict", ckpt)
    sd = extract_sub_state_dict(sd, "backbone.")
    print_state_dict(sd)

    cfg = load_config("./config.yaml")

    backbone = CSPNext(cfg=cfg)

    param_name = "stem.0.conv.weight"

    before = backbone.state_dict()[param_name].clone()

    backbone.load_state_dict(sd, strict=False)

    after = backbone.state_dict()[param_name]

    print(torch.allclose(before, after))  # False se i pesi sono cambiati

    x = torch.randn(1, 3, 640, 640)
    y = backbone(x)
    print(y.shape)


if __name__ == "__main__":
    main()
