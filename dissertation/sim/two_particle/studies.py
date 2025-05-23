from abc import ABC, abstractmethod
from functools import lru_cache
from typing import ClassVar, Sequence
from pathlib import Path
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict
import itertools

from dissertation.config import in_build_dir
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
    duration=3.6e3,
)


class StudyBase(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    KEY: ClassVar[str]
    INSTANCES: ClassVar[Sequence["StudyBase"]]

    @property
    def dir(self) -> Path:
        return in_build_dir(THIS_DIR / "studies") / self.key

    @classmethod
    @property
    def DIR(cls) -> Path:
        return in_build_dir(THIS_DIR / "studies" / cls.KEY)

    @property
    @abstractmethod
    def key(self) -> str:
        """str representation for key indexing"""

    @property
    @abstractmethod
    def display(self) -> str:
        """str representation for display in plot"""

    @property
    @abstractmethod
    def input(self) -> Input:
        """return respective input instance"""

    @property
    @abstractmethod
    def line_style(self) -> dict:
        """return repective line style for plot"""


class TimeStepStudy(StudyBase):
    KEY: ClassVar[str] = "time_step"

    LIMITS: ClassVar[list[float]] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    COLORS: ClassVar[dict[float, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    step_angle_limit: float

    @property
    def key(self) -> str:
        return f"{TimeStepStudy.KEY}/{self.step_angle_limit}"

    @property
    def display(self) -> str:
        return rf"{self.step_angle_limit}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.free_surface_remesher_options = None
        model.time_step_angle_limit = self.step_angle_limit
        return model

    @property
    def line_style(self) -> dict:
        return dict(color=TimeStepStudy.COLORS[self.step_angle_limit])


TimeStepStudy.INSTANCES = [TimeStepStudy(step_angle_limit=lim) for lim in TimeStepStudy.LIMITS]


class SurfaceRemeshingStudy(StudyBase):
    KEY: ClassVar[str] = "surface_remeshing"

    NODE_COUNTS: ClassVar[list[int]] = [50, 100, 200]
    LIMITS: ClassVar[list[float | None]] = [None, 0.01, 0.02, 0.05]
    NODE_COUNT_STYLES: ClassVar[dict[float | None, str]] = {
        n: s for n, s in zip(NODE_COUNTS, ["dashed", "solid", "dotted"])
    }
    LIMIT_COLORS: ClassVar[dict[float | None, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    node_count: int
    limit: float | None

    @property
    def key(self) -> str:
        return f"{SurfaceRemeshingStudy.KEY}/{self.node_count}_{self.limit}"

    @property
    def display(self) -> str:
        return f"n = {self.node_count}, limit = {self.limit}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.particle1.node_count = self.node_count
        model.particle2.node_count = self.node_count
        model.free_surface_remesher_options = (
            FreeSurfaceRemesherOptions(deletion_limit=self.limit) if self.limit else None
        )
        return model

    @property
    def line_style(self) -> dict:
        return dict(
            linestyle=SurfaceRemeshingStudy.NODE_COUNT_STYLES[self.node_count],
            color=SurfaceRemeshingStudy.LIMIT_COLORS[self.limit],
        )


SurfaceRemeshingStudy.INSTANCES = [
    SurfaceRemeshingStudy(node_count=node_count, limit=limit)
    for node_count, limit in itertools.product(SurfaceRemeshingStudy.NODE_COUNTS, SurfaceRemeshingStudy.LIMITS)
]


class NeckRemeshingStudy(StudyBase):
    KEY: ClassVar[str] = "neck_remeshing"

    NODE_COUNTS: ClassVar[list[int]] = [50, 100, 200]
    LIMITS: ClassVar[list[float]] = [0.1, 0.3, 0.5, 0.7]
    NODE_COUNT_STYLES: ClassVar[dict[float, str]] = {n: s for n, s in zip(NODE_COUNTS, ["dashed", "solid", "dotted"])}
    LIMIT_COLORS: ClassVar[dict[float | None, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    node_count: int
    limit: float

    @property
    def key(self) -> str:
        return f"{NeckRemeshingStudy.KEY}/{self.node_count}_{self.limit}"

    @property
    def display(self) -> str:
        return f"n = {self.node_count}, limit = {self.limit}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.particle1.node_count = self.node_count
        model.particle2.node_count = self.node_count
        model.free_surface_remesher_options = None
        model.neck_deletion_limit = self.limit
        return model

    @property
    def line_style(self) -> dict:
        return dict(
            linestyle=NeckRemeshingStudy.NODE_COUNT_STYLES[self.node_count],
            color=NeckRemeshingStudy.LIMIT_COLORS[self.limit],
        )


NeckRemeshingStudy.INSTANCES = [
    NeckRemeshingStudy(node_count=node_count, limit=limit)
    for node_count, limit in itertools.product(NeckRemeshingStudy.NODE_COUNTS, NeckRemeshingStudy.LIMITS)
]

STUDIES = [TimeStepStudy, SurfaceRemeshingStudy, NeckRemeshingStudy]
