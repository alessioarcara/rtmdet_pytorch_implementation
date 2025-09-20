from typing import TypeAlias, Dict
import torch

StateDict: TypeAlias = Dict[str, torch.Tensor]


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
