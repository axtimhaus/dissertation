from abc import abstractmethod, ABC
from pathlib import Path
from typing import Literal, Iterable
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

THIS_DIR = Path(__file__).parent


class ParticleInput(BaseModel):
    id: UUID
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
    gas_constant: float
    temperature: float
    duration: float
    vacancy_concentration: float

    @property
    def _time_norm_common(self):
        return self.gas_constant * self.temperature / self.vacancy_concentration * self.particle1.radius ** 4 /  self.material1.molar_mass * self.material1.density

    @property
    def time_norm_surface(self):
        return self._time_norm_common / (self.material1.surface.diffusion_coefficient * self.material1.surface.energy)

    @property
    def time_norm_grain_boundary(self):
        return self._time_norm_common / (self.grain_boundary.diffusion_coefficient * self.grain_boundary.energy)


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

    def dir(self, value: float | str | None = None):
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
