from pathlib import Path
from uuid import uuid5

import numpy as np
from pydantic import BaseModel, ConfigDict

from dissertation.config import ROOT_NAMESPACE_UUID, in_build_dir
from dissertation.sim.packings.input import (
    Input,
    InterfaceInput,
    MaterialInput,
    ParticleInput,
)

THIS_DIR = Path(__file__).parent

NAMESPACE = uuid5(ROOT_NAMESPACE_UUID, "sim/packings")


def create_particle(index: int, x: float, y: float, rotation_angle: float = 0):
    radius = 100e-6
    distance = 1.99 * radius
    return ParticleInput(
        id=uuid5(NAMESPACE, f"particle{index}"),
        x=x * distance,
        y=y * distance,
        rotation_angle=rotation_angle,
        radius=radius,
        node_count=200,
    )


BASE_SURFACE = InterfaceInput(energy=0.9, diffusion_coefficient=1.65e-14)
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
    particles=[],
    material=BASE_MATERIAL.model_copy(deep=True),
    grain_boundary=BASE_GRAIN_BOUNDARY.model_copy(deep=True),
    gas_constant=8.31446261815324,
    temperature=1273,
    duration=3.6e4,
)


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    display: str
    input: Input
    line_style: dict

    @property
    def dir(self) -> Path:
        return in_build_dir(THIS_DIR / "cases") / self.key


_COMMON_STYLE = dict()

PAIR_INPUT = BASE_INPUT.model_copy(deep=True)
PAIR_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, 1, 0),
]
PAIR_CASE = Case(
    key="pair",
    display="Pair",
    input=PAIR_INPUT,
    line_style=_COMMON_STYLE | dict(color="C0"),
)
PAIR_CASE_INERT = Case(
    key="pair_inert",
    display="Pair with Inert",
    input=PAIR_INPUT.model_copy(update=dict(inert_particle_id=1)),
    line_style=_COMMON_STYLE | dict(color="C0", linestyle="dashed"),
)

TRIANGLE_INPUT = BASE_INPUT.model_copy(deep=True)
TRIANGLE_INPUT.particles = [
    create_particle(0, 0, 0, np.deg2rad(30)),
    create_particle(1, 1, 0, np.deg2rad(150)),
    create_particle(2, 0.5, np.sqrt(3) / 2, np.deg2rad(270)),
]
TRIANGLE_CASE = Case(
    key="triangle",
    display="Triangle",
    input=TRIANGLE_INPUT,
    line_style=_COMMON_STYLE | dict(color="C1"),
)
TRIANGLE_CASE_INERT = Case(
    key="triangle_inert",
    display="Triangle with Inert",
    input=TRIANGLE_INPUT.model_copy(update=dict(inert_particle_id=2)),
    line_style=_COMMON_STYLE | dict(color="C1", linestyle="dashed"),
)

SQUARE_INPUT = BASE_INPUT.model_copy(deep=True)
SQUARE_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, 1, 0),
    create_particle(2, 1, 1),
    create_particle(3, 0, 1),
]
SQUARE_CASE = Case(
    key="square",
    display="Square",
    input=SQUARE_INPUT,
    line_style=_COMMON_STYLE
    | dict(
        color="C2",
    ),
)
SQUARE_CASE_INERT = Case(
    key="square_inert",
    display="Square with Inert",
    input=SQUARE_INPUT.model_copy(update=dict(inert_particle_id=2)),
    line_style=_COMMON_STYLE
    | dict(
        color="C2",
        linestyle="dashed",
    ),
)

RHOMBUS_ANGLE = np.deg2rad(70)
_RHOMBUS_LARGE_DIAG = 2 * np.cos(RHOMBUS_ANGLE / 2)
_RHOMBUS_SMALL_DIAG = 2 * np.sin(RHOMBUS_ANGLE / 2)
RHOMBUS_INPUT = BASE_INPUT.model_copy(deep=True)
RHOMBUS_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, _RHOMBUS_LARGE_DIAG / 2, -_RHOMBUS_SMALL_DIAG / 2, np.deg2rad(90)),
    create_particle(2, _RHOMBUS_LARGE_DIAG, 0, np.deg2rad(180)),
    create_particle(3, _RHOMBUS_LARGE_DIAG / 2, _RHOMBUS_SMALL_DIAG / 2, np.deg2rad(270)),
]
RHOMBUS_CASE = Case(
    key="rhombus",
    display="Rhombus",
    input=RHOMBUS_INPUT,
    line_style=_COMMON_STYLE
    | dict(
        color="C3",
    ),
)
RHOMBUS_CASE_INERT = Case(
    key="rhombus_inert",
    display="Rhombus with Inert",
    input=RHOMBUS_INPUT.model_copy(update=dict(inert_particle_id=3)),
    line_style=_COMMON_STYLE
    | dict(
        color="C3",
        linestyle="dashed",
    ),
)

CASES = [
    PAIR_CASE,
    TRIANGLE_CASE,
    SQUARE_CASE,
    RHOMBUS_CASE,
    PAIR_CASE_INERT,
    TRIANGLE_CASE_INERT,
    SQUARE_CASE_INERT,
    RHOMBUS_CASE_INERT,
]
