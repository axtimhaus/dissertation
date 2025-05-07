from abc import ABC
from pathlib import Path
from uuid import UUID
import itertools

import numpy as np
from pydantic import BaseModel, ConfigDict

from dissertation.sim.two_particle.input import (
    Input,
    InterfaceInput,
    MaterialInput,
    ParticleInput,
    FreeSurfaceRemesherOptions,
)

THIS_DIR = Path(__file__).parent

PARTICLE1_ID = UUID("989b9875-2a6b-40c3-ab2f-5ebc96682dbe")
PARTICLE2_ID = UUID("10cac1cc-6205-4b91-85b4-4e9d6f126274")

BASE_PARTICLE = ParticleInput(id=PARTICLE1_ID, radius=100e-6)

BASE_SURFACE = InterfaceInput(energy=0.9, diffusion_coefficient=1.65e-10)
BASE_GRAIN_BOUNDARY = InterfaceInput(
    energy=BASE_SURFACE.energy * 0.5,
    diffusion_coefficient=BASE_SURFACE.diffusion_coefficient,
)

BASE_MATERIAL = MaterialInput(
    surface=BASE_SURFACE.model_copy(),
    density=1.8e3,
    molar_mass=101.96e-3,
)

BASE_INPUT = Input(
    particle1=BASE_PARTICLE.model_copy(deep=True),
    particle2=BASE_PARTICLE.model_copy(
        deep=True,
        update={
            "id": PARTICLE2_ID,
            "x": 1.99 * BASE_PARTICLE.radius,
            "rotation_angle": np.pi,
        },
    ),
    material1=BASE_MATERIAL.model_copy(deep=True),
    material2=BASE_MATERIAL.model_copy(deep=True),
    grain_boundary=BASE_GRAIN_BOUNDARY.model_copy(deep=True),
    gas_constant=8.31446261815324,
    temperature=1273,
    vacancy_concentration=1e-4,
    duration=3.6e5,
)


def get_base_input_copy():
    return BASE_INPUT.model_copy(deep=True)


class RemeshingStudy(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    node_count: int
    surface_remesher_limit: float | None

    def __str__(self) -> str:
        return f"{self.node_count}_{self.surface_remesher_limit}"

    def dir(self):
        base = THIS_DIR / "runs" / str(self)
        return base

    @property
    def input(self):
        model = BASE_INPUT.model_copy(deep=True)
        model.particle1.node_count = self.node_count
        model.particle2.node_count = self.node_count
        model.free_surface_remesher_options = (
            FreeSurfaceRemesherOptions(addition_limit=self.surface_remesher_limit)
            if self.surface_remesher_limit
            else None
        )
        return model


STUDIES = [
    RemeshingStudy(node_count=node_count, surface_remesher_limit=limit)
    for node_count, limit in itertools.product([50, 100, 200], [None, 0.01, 0.02, 0.05])
]
