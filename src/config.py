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
    neck_out_channels: int = Field(
        ...,
        description="Number of channels of the output convolution layers in the PAFPN module",
    )
    head_num_stacked_convs: int = Field(
        ...,
        description="Number of convolution blocks in each classification/regression tower",
    )
    head_num_levels: int = Field(
        ...,
        ge=1,
        description="Number of pyramid levels the head operates on (e.g., 3 for P3-P5). Must equal the number of feature maps provided by the neck",
    )


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).expanduser().read_text()))
