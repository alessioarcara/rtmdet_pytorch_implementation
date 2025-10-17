from typing import Dict, Literal, TypeAlias

import torch

StateDict: TypeAlias = Dict[str, torch.Tensor]
PresetName: TypeAlias = Literal["tiny", "small", "medium", "large"]
