from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals

from rtmdet.typings import StateDict

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


def _safe_load_state_dict(model: torch.nn.Module, sd: StateDict) -> None:
    """
    Load a state_dict into a model, ignoring any keys with mismatched shapes or missing entries.
    """
    model_sd: StateDict = model.state_dict()
    compatible_weights = {
        k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape
    }

    model.load_state_dict(compatible_weights, strict=False)


def load_and_verify_weight(model: torch.nn.Module, sd: StateDict) -> None:
    before_sd = {k: v.clone() for k, v in model.state_dict().items()}

    _safe_load_state_dict(model, sd)

    after_sd = model.state_dict()

    for name, before in before_sd.items():
        after = after_sd[name]
        if torch.allclose(before, after):
            print(f"🔴 {name}: unchanged")
        else:
            print(f"🟢 {name}: updated")


def _default_cache_dir() -> Path:
    cache_dir = Path(torch.hub.get_dir()) / "rtmdet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cached_weights_path(url: str) -> Path:
    return _default_cache_dir() / Path(url).name


def _download_if_needed(url: str) -> Path:
    dst = _cached_weights_path(url)

    if dst.exists():
        return dst

    torch.hub.download_url_to_file(url, dst)
    return dst
