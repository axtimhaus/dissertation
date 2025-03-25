from abc import ABC
from pathlib import Path
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict

from dissertation.sim.two_particle.input import (
    Input,
    InterfaceInput,
    MaterialInput,
    ParticleInput, FreeSurfaceRemesherOptions,
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

    name: str
    display_tex: str = ""
    input: Input

    def __str__(self) -> str:
        return f"{self.name}"

    def dir(self):
        base = THIS_DIR / "runs" / str(self)
        return base

def with_node_count(model: Input, n, **update):
    model = model.model_copy(deep=True, update=update)
    model.particle1.node_count = n
    model.particle2.node_count = n
    return model

STUDIES = [
    RemeshingStudy(
        name="no_surface_remeshing_50",
        input=with_node_count(BASE_INPUT, 50, time_step_angle_limit=0.02),
        display_tex="No Surface Remeshing 50",
    ),
    RemeshingStudy(
        name="no_surface_remeshing",
        input=BASE_INPUT,
        display_tex="No Surface Remeshing 100",
    ),
    # RemeshingStudy(
    #     name="no_surface_adding",
    #     input=BASE_INPUT.model_copy(update=dict(free_surface_remesher_options=FreeSurfaceRemesherOptions(addition_limit=1000000))),
    #     display_tex="No Surface Node Adding",
    # ),
    RemeshingStudy(
        name="default",
        input=BASE_INPUT.model_copy(update=dict(free_surface_remesher_options=FreeSurfaceRemesherOptions())),
        display_tex="Default Parameters",
    ),
    RemeshingStudy(
        name="default_200",
        input=with_node_count(BASE_INPUT, 200, free_surface_remesher_options=FreeSurfaceRemesherOptions()),
        display_tex="Default Parameters 200",
    ),
    RemeshingStudy(
        name="limit_005",
        input=BASE_INPUT.model_copy(update=dict(free_surface_remesher_options=FreeSurfaceRemesherOptions(deletion_limit=0.05))),
        display_tex="Deletion Limit = 0.05",
    ),
]
