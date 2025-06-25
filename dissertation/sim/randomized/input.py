from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
from pydantic import BaseModel, Field
from dissertation.config import ROOT_NAMESPACE_UUID

THIS_DIR = Path(__file__).parent
NAMESPACE = uuid5(ROOT_NAMESPACE_UUID, "sim/randomized")

GAS_CONSTANT = 8.31446261815324
TEMPERATURE = 1273
DURATION = 3.6e3


class InterfaceInput(BaseModel):
    energy: float = Field(gt=0)
    diffusion_coefficient: float = Field(gt=0)


class MaterialInput(BaseModel):
    surface: InterfaceInput
    density: float = Field(gt=0)
    molar_mass: float = Field(gt=0)


class ParticleInput(BaseModel):
    id: UUID
    x: float = 0
    y: float = 0
    rotation_angle: float = Field(ge=0, lt=2 * np.pi, default=0)
    radius: float = Field(ge=0)
    ovality: float = Field(ge=0, lt=1, default=1)
    peak_count: int = Field(ge=0, default=0)
    peak_height: float = Field(ge=0, lt=1, default=0)
    node_count: int = Field(gt=0, default=200)
    material: MaterialInput
    grain_boundaries: dict[UUID, InterfaceInput] = {}


REFERENCE_MATERIAL = MaterialInput(
    surface=InterfaceInput(energy=0.9, diffusion_coefficient=1.65e-14),
    density=1.8e3,
    molar_mass=101.96e-3,
)
REFERENCE_PARTICLE = ParticleInput(
    id=uuid5(NAMESPACE, "reference_particle"), radius=100e-6, material=REFERENCE_MATERIAL
)


class Input(BaseModel):
    particles: list[ParticleInput]
    gas_constant: float = GAS_CONSTANT
    temperature: float = TEMPERATURE
    duration: float = DURATION

    @property
    def time_norm_surface(self):
        return (
            self.gas_constant
            * self.temperature
            * REFERENCE_PARTICLE.radius**4
            / REFERENCE_MATERIAL.molar_mass
            * REFERENCE_MATERIAL.density
            / (REFERENCE_MATERIAL.surface.diffusion_coefficient * REFERENCE_MATERIAL.surface.energy)
        )
