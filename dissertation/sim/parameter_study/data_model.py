from abc import abstractmethod, ABC
from pathlib import Path
from typing import Literal, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from numpy.typing import ArrayLike

THIS_DIR = Path(__file__).parent


class ParticleInput(BaseModel):
    x: float = 0
    y: float = 0
    radius: float = Field(ge=0)
    ovality: float = Field(ge=0, lt=1, default=0)
    peak_count: int = Field(ge=0, default=0)
    peak_height: float = Field(ge=0, lt=1, default=0)
    node_count: int = Field(gt=0, default=50)


class InterfaceInput(BaseModel):
    energy: float = Field(gt=0)
    diffusion_coefficient: float = Field(gt=0)


class MaterialInput(BaseModel):
    surface: InterfaceInput
    density: float = Field(gt=0)
    molar_mass: float = Field(gt=0)


class Input(BaseModel):
    particle1: ParticleInput
    particle2: ParticleInput
    material1: MaterialInput
    material2: MaterialInput
    grain_boundary: InterfaceInput


class ParameterStudy(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    parameter_name: str
    min: float = 0
    max: float = 10
    count: int = 10
    scale: Literal["lin", "log", "geom"]
    display_tex: str = ""

    def __str__(self) -> str:
        return f"{self.parameter_name}_{self.min}_{self.max}_{self.scale}_{self.count}"

    def dir(self, value: float | None = None):
        base = THIS_DIR / "runs" / str(self)
        if value is not None:
            return base / str(value)
        return base

    @property
    def parameter_values(self) -> Iterable[float]:
        if self.scale == "lin":
            return np.linspace(self.min, self.max, self.count, True)

        if self.scale == "geom":
            return np.geomspace(self.min, self.max, self.count, True)

        if self.scale == "log":
            return np.logspace(self.min, self.max, self.count, True)

        raise ValueError(self.scale)

    @abstractmethod
    def input_for(self, parameter_value: float) -> Input:
        raise NotImplementedError()
