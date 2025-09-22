from typing import Dict, TypeAlias

import numpy as np
import torch
from torch.serialization import add_safe_globals

StateDict: TypeAlias = Dict[str, torch.Tensor]


HistoryBufferDummy = type("HistoryBuffer", (), {})
HistoryBufferDummy.__module__ = "mmengine.logging.history_buffer"


def load_mmdet_checkpoint(path: str, map_location: str = "cpu") -> StateDict:
    add_safe_globals(
        [
            HistoryBufferDummy,
            np.dtype,
            np.core.multiarray.scalar,  # type: ignore
            np.core.multiarray._reconstruct,  # type: ignore
            np.ndarray,
            np.float64,
            np.dtypes.Float64DType,
            np.dtypes.Int64DType,
        ]
    )

    ckpt = torch.load(path, map_location=map_location, weights_only=True)

    state_dict = ckpt.get("state_dict", ckpt)
    return state_dict


def extract_sub_state_dict(sd: StateDict, prefix: str) -> StateDict:
    sub_state_dict = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            sub_state_dict[k[len(prefix) :]] = v
    return sub_state_dict


def print_state_dict(sd: StateDict, max_key_len: int = 60) -> None:
    for k, v in sd.items():
        key_str = k.ljust(max_key_len)
        if isinstance(v, torch.Tensor):
            print(f"{key_str} {tuple(v.shape)}")
        else:
            print(f"{key_str} ({type(v).__name__})")


def check_params_updated(model: torch.nn.Module, sd: StateDict) -> None:
    before_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(sd, strict=False)
    after_sd = model.state_dict()

    for name, before in before_sd.items():
        after = after_sd[name]
        if torch.allclose(before, after):
            print(f"🔴 {name}: unchanged")
        else:
            print(f"🟢 {name}: updated")
