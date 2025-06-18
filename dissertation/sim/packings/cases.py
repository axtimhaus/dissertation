import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import P_PID
from pathlib import Path
from typing import ClassVar, Literal
from uuid import UUID, uuid5

import matplotlib
import matplotlib.colors
import numpy as np
from pydantic import BaseModel, ConfigDict

from dissertation.config import in_build_dir, ROOT_NAMESPACE_UUID
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
    particles=[],
    material=BASE_MATERIAL.model_copy(deep=True),
    grain_boundary=BASE_GRAIN_BOUNDARY.model_copy(deep=True),
    gas_constant=8.31446261815324,
    temperature=1273,
    vacancy_concentration=1e-4,
    duration=3.6e3,
)


class Case(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    key: str
    display: str
    input: Input
    line_style: dict

    @property
    def dir(self) -> Path:
        return in_build_dir(THIS_DIR / "cases") / self.key


_COMMON_STYLE = dict(linewidth=1)

PAIR_INPUT = BASE_INPUT.model_copy(deep=True)
PAIR_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, 1, 0),
]
PAIR_CASE = Case(key="pair", display="Pair", input=PAIR_INPUT, line_style=_COMMON_STYLE | dict(color="C0"))

TRIANGLE_INPUT = BASE_INPUT.model_copy(deep=True)
TRIANGLE_INPUT.particles = [
    create_particle(0, 0, 0, np.deg2rad(30)),
    create_particle(1, 1, 0, np.deg2rad(150)),
    create_particle(2, 0.5, np.sqrt(3) / 2, np.deg2rad(270)),
]
TRIANGLE_CASE = Case(
    key="triangle", display="Triangle", input=TRIANGLE_INPUT, line_style=_COMMON_STYLE | dict(color="C1")
)

SQUARE_INPUT = BASE_INPUT.model_copy(deep=True)
SQUARE_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, 1, 0),
    create_particle(2, 1, 1),
    create_particle(3, 0, 1),
]
SQUARE_CASE = Case(key="square", display="Square", input=SQUARE_INPUT, line_style=_COMMON_STYLE | dict(color="C2"))

RHOMBUS_ANGLE = np.deg2rad(80)
_RHOMBUS_LARGE_DIAG = 2 * np.cos(RHOMBUS_ANGLE / 2)
_RHOMBUS_SMALL_DIAG = 2 * np.sin(RHOMBUS_ANGLE / 2)
RHOMBUS_INPUT = BASE_INPUT.model_copy(deep=True)
RHOMBUS_INPUT.particles = [
    create_particle(0, 0, 0),
    create_particle(1, _RHOMBUS_LARGE_DIAG / 2, -_RHOMBUS_SMALL_DIAG / 2, np.deg2rad(90)),
    create_particle(2, _RHOMBUS_LARGE_DIAG, 0, np.deg2rad(180)),
    create_particle(3, _RHOMBUS_LARGE_DIAG / 2, _RHOMBUS_SMALL_DIAG / 2, np.deg2rad(270)),
]
RHOMBUS_CASE = Case(key="rhombus", display="Rhombus", input=RHOMBUS_INPUT, line_style=_COMMON_STYLE | dict(color="C3"))

CASES = [
    PAIR_CASE,
    TRIANGLE_CASE,
    SQUARE_CASE,
    RHOMBUS_CASE,
]
