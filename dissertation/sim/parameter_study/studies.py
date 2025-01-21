import deepdiff
from uuid import UUID

from dissertation.sim.parameter_study.data_model import (
    ParameterStudy,
    Input,
    ParticleInput,
    MaterialInput,
    InterfaceInput,
)


def hash(studies: list[ParameterStudy]) -> str:
    return deepdiff.DeepHash(studies)[studies]

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
    particle2=BASE_PARTICLE.model_copy(deep=True, update={"id": PARTICLE2_ID, "x": 1.99 * BASE_PARTICLE.radius}),
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


class ParticleSizeRatioStudy(ParameterStudy):
    def input_for(self, parameter_value: float) -> Input:
        model = get_base_input_copy()
        model.particle2.radius *= parameter_value
        model.particle2.x = (model.particle1.radius * 0.99 + model.particle2.radius)
        model.particle2.node_count = int(model.particle2.node_count * parameter_value)

        return model


class SurfaceBoundaryEnergyStudy(ParameterStudy):
    def input_for(self, parameter_value: float) -> Input:
        model = get_base_input_copy()
        model.grain_boundary.energy *= model.material1.surface.energy * parameter_value
        return model

class SurfaceBoundaryDiffusionStudy(ParameterStudy):
    def input_for(self, parameter_value: float) -> Input:
        model = get_base_input_copy()
        model.grain_boundary.diffusion_coefficient *= model.material1.surface.diffusion_coefficient * parameter_value
        return model


STUDIES = [
    ParticleSizeRatioStudy(
        parameter_name="particle_size_ratio",
        min=1,
        max=10,
        count=10,
        scale="geom",
    ),
    SurfaceBoundaryEnergyStudy(
        parameter_name="surface_boundary_energy",
        min=0.01,
        max=1,
        count=10,
        scale="geom"
    ),
    SurfaceBoundaryDiffusionStudy(
        parameter_name="surface_boundary_diffusion",
        min=0.01,
        max=1,
        count=10,
        scale="geom"
    ),
]
