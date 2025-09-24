from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    deepen_factor: float = Field(
        ...,
        description="Scaling factor for the model depth (e.g., the number of layers or blocks)",
    )
    widen_factor: float = Field(
        ...,
        description="Scaling factor for the model width (e.g., the number of channels or neurons)",
    )
    pafpn_out_channels: int = Field(
        ...,
        description="Output channels of the convolution layers in the PAFPN module",
    )


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).expanduser().read_text()))
