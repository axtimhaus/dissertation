import deepdiff

from dissertation.sim.parameter_study.data_model import (
    ParameterStudy,
    Input,
    ParticleInput,
    MaterialInput,
    InterfaceInput,
)


def hash(studies: list[ParameterStudy]) -> str:
    return deepdiff.DeepHash(studies)[studies]


BASE_PARTICLE = ParticleInput(radius=100e-6)

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
    particle2=BASE_PARTICLE.model_copy(deep=True, update={"x": 210e-6}),
    material1=BASE_MATERIAL.model_copy(deep=True),
    material2=BASE_MATERIAL.model_copy(deep=True),
    grain_boundary=BASE_GRAIN_BOUNDARY.model_copy(deep=True),
)


def get_base_input_copy():
    return BASE_INPUT.model_copy(deep=True)


class ParticleSizeRatioStudy(ParameterStudy):
    def input_for(self, parameter_value: float) -> Input:
        model = get_base_input_copy()
        model.particle2.radius *= parameter_value
        model.particle2.x += (model.particle1.radius + model.particle2.radius) * 1.05
        return model


STUDIES = [
    ParticleSizeRatioStudy(
        parameter_name="particle_size_ratio",
        min=1,
        max=10,
        count=10,
        scale="geom",
    ),
]
