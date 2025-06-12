from pathlib import Path
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).parent


class ParticleInput(BaseModel):
    id: UUID
    x: float = 0
    y: float = 0
    rotation_angle: float = Field(ge=0, lt=2 * np.pi, default=0)
    radius: float = Field(ge=0)
    ovality: float = Field(ge=0, lt=1, default=1)
    peak_count: int = Field(ge=0, default=0)
    peak_height: float = Field(ge=0, lt=1, default=0)
    node_count: int = Field(gt=0, default=100)


class InterfaceInput(BaseModel):
    energy: float = Field(gt=0)
    diffusion_coefficient: float = Field(gt=0)


class MaterialInput(BaseModel):
    surface: InterfaceInput
    density: float = Field(gt=0)
    molar_mass: float = Field(gt=0)


class Input(BaseModel):
    particles: list[ParticleInput]
    material: MaterialInput
    grain_boundary: InterfaceInput
    gas_constant: float
    temperature: float
    duration: float
    vacancy_concentration: float

    @property
    def _time_norm_common(self):
        return (
            self.gas_constant
            * self.temperature
            / self.vacancy_concentration
            * self.particles[0].radius**4
            / self.material.molar_mass
            * self.material.density
        )

    @property
    def time_norm_surface(self):
        return self._time_norm_common / (self.material.surface.diffusion_coefficient * self.material.surface.energy)

    @property
    def time_norm_grain_boundary(self):
        return self._time_norm_common / (self.grain_boundary.diffusion_coefficient * self.grain_boundary.energy)
