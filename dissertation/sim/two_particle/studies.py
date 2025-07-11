import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal
from uuid import UUID

import matplotlib
import matplotlib.colors
import numpy as np
from pydantic import BaseModel, ConfigDict

from dissertation.config import in_build_dir
from dissertation.sim.two_particle.input import (
    FreeSurfaceRemesherOptions,
    Input,
    InterfaceInput,
    MaterialInput,
    ParticleInput,
)

THIS_DIR = Path(__file__).parent

PARTICLE1_ID = UUID("989b9875-2a6b-40c3-ab2f-5ebc96682dbe")
PARTICLE2_ID = UUID("10cac1cc-6205-4b91-85b4-4e9d6f126274")

BASE_PARTICLE = ParticleInput(id=PARTICLE1_ID, radius=100e-6)

BASE_SURFACE = InterfaceInput(energy=0.9, diffusion_coefficient=1.65e-14)
BASE_GRAIN_BOUNDARY = InterfaceInput(
    energy=BASE_SURFACE.energy * 0.5 / 2,
    diffusion_coefficient=BASE_SURFACE.diffusion_coefficient,
)

BASE_MATERIAL = MaterialInput(
    surface=BASE_SURFACE.model_copy(),
    grain_boundary=BASE_GRAIN_BOUNDARY.model_copy(deep=True),
    density=1.8e3,
    molar_mass=101.96e-3,
)

INTRUSION = BASE_PARTICLE.radius * 0.01

BASE_INPUT = Input(
    particle1=BASE_PARTICLE.model_copy(deep=True),
    particle2=BASE_PARTICLE.model_copy(
        deep=True,
        update={
            "id": PARTICLE2_ID,
            "x": 2 * BASE_PARTICLE.radius - INTRUSION,
            "rotation_angle": np.pi,
        },
    ),
    material1=BASE_MATERIAL.model_copy(deep=True),
    material2=BASE_MATERIAL.model_copy(deep=True),
    gas_constant=8.31446261815324,
    temperature=1273,
    duration=3.6e3,
)


class StudyBase(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    KEY: ClassVar[str]
    TITLE: ClassVar[str]
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
    KEY = "time_step"
    TITLE = "Max. Displacement Angle"

    LIMITS: ClassVar[list[float]] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    COLORS: ClassVar[dict[float, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    step_angle_limit: float

    @property
    def key(self) -> str:
        return f"{TimeStepStudy.KEY}/{self.step_angle_limit:.3f}"

    @property
    def display(self) -> str:
        return rf"{self.step_angle_limit:.3g}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.free_surface_remesher_options = None
        model.time_step_angle_limit = self.step_angle_limit
        return model

    @property
    def line_style(self) -> dict:
        return dict(
            linewidth=1,
            color=TimeStepStudy.COLORS[self.step_angle_limit],
        )


TimeStepStudy.INSTANCES = [TimeStepStudy(step_angle_limit=lim) for lim in TimeStepStudy.LIMITS]


class SurfaceRemeshingStudy(StudyBase):
    KEY = "surface_remeshing"
    TITLE = "Node Count / Surface Remeshing Limit"

    NODE_COUNTS: ClassVar[list[int]] = [50, 100, 200]
    LIMITS: ClassVar[list[float]] = [0.00, 0.01, 0.02, 0.05]
    NODE_COUNT_STYLES: ClassVar[dict[float | None, str]] = {
        n: s for n, s in zip(NODE_COUNTS, ["dashed", "solid", "dotted"], strict=False)
    }
    LIMIT_COLORS: ClassVar[dict[float | None, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    node_count: int
    limit: float | None

    @property
    def key(self) -> str:
        return f"{SurfaceRemeshingStudy.KEY}/{self.node_count}_{self.limit:.2f}"

    @property
    def display(self) -> str:
        return f"{self.node_count}/{self.limit:.2g}"

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
            linewidth=1,
            linestyle=SurfaceRemeshingStudy.NODE_COUNT_STYLES[self.node_count],
            color=SurfaceRemeshingStudy.LIMIT_COLORS[self.limit],
        )


SurfaceRemeshingStudy.INSTANCES = [
    SurfaceRemeshingStudy(node_count=node_count, limit=limit)
    for node_count, limit in itertools.product(SurfaceRemeshingStudy.NODE_COUNTS, SurfaceRemeshingStudy.LIMITS)
]


class NeckRemeshingStudy(StudyBase):
    KEY = "neck_remeshing"
    TITLE = "Node Count / Neck Remeshing Limit"

    NODE_COUNTS: ClassVar[list[int]] = [50, 100, 200]
    LIMITS: ClassVar[list[float]] = [0.3, 0.5, 0.7]
    NODE_COUNT_STYLES: ClassVar[dict[float, str]] = {
        n: s for n, s in zip(NODE_COUNTS, ["dashed", "solid", "dotted"], strict=False)
    }
    LIMIT_COLORS: ClassVar[dict[float | None, str]] = {lim: f"C{i}" for i, lim in enumerate(LIMITS)}

    node_count: int
    limit: float

    @property
    def key(self) -> str:
        return f"{NeckRemeshingStudy.KEY}/{self.node_count}_{self.limit:.1f}"

    @property
    def display(self) -> str:
        return f"{self.node_count}/{self.limit:.1g}"

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
            linewidth=1,
            linestyle=NeckRemeshingStudy.NODE_COUNT_STYLES[self.node_count],
            color=NeckRemeshingStudy.LIMIT_COLORS[self.limit],
        )


NeckRemeshingStudy.INSTANCES = [
    NeckRemeshingStudy(node_count=node_count, limit=limit)
    for node_count, limit in itertools.product(NeckRemeshingStudy.NODE_COUNTS, NeckRemeshingStudy.LIMITS)
]


class LongRunStudy(StudyBase):
    KEY = "long_run"
    TITLE = "Long Run"

    @property
    def key(self) -> str:
        return f"{LongRunStudy.KEY}/run"

    @property
    def display(self) -> str:
        return f"{LongRunStudy.KEY}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.particle1.node_count = 200
        model.particle2.node_count = 200
        model.time_step_angle_limit = 0.005
        model.free_surface_remesher_options = FreeSurfaceRemesherOptions(deletion_limit=0.05)
        model.neck_deletion_limit = 0.5
        model.duration = 3.6e3 * 1e3
        return model

    @property
    def line_style(self) -> dict:
        return dict(
            linewidth=1,
        )


LongRunStudy.INSTANCES = [LongRunStudy()]


class DimlessParameterStudy(StudyBase, ABC):
    CMAP: ClassVar[matplotlib.colors.Colormap] = matplotlib.colormaps["viridis"]

    MIN: ClassVar[float]
    MAX: ClassVar[float]
    COUNT: ClassVar[int] = 11
    SCALE: ClassVar[Literal["lin", "log", "geom"]]

    value: float

    @property
    def key(self) -> str:
        return f"{self.KEY}/{self.value:.5f}"

    @property
    def display(self) -> str:
        return f"{self.value:.5g}"

    @property
    def input(self) -> Input:
        model = BASE_INPUT.model_copy(deep=True)
        model.particle1.node_count = 200
        model.particle2.node_count = 200
        model.time_step_angle_limit = 0.005
        model.free_surface_remesher_options = FreeSurfaceRemesherOptions(deletion_limit=0.05)
        model.neck_deletion_limit = 0.5
        return model

    @property
    def line_style(self) -> dict:
        if self.SCALE == "lin" or self.SCALE == "log":
            return dict(
                linewidth=1,
                color=type(self).CMAP((self.value - self.MIN) / (self.MAX - self.MIN)),
            )
        if self.SCALE == "geom":
            return dict(
                linewidth=1,
                color=type(self).CMAP((np.log(self.value) - np.log(self.MIN)) / (np.log(self.MAX) - np.log(self.MIN))),
            )
        raise ValueError()

    @property
    def real_value(self):
        if self.SCALE == "lin" or self.SCALE == "geom":
            return self.value
        if self.SCALE == "log":
            return np.exp(self.value)
        raise ValueError()

    @classmethod
    @property
    def values(cls) -> np.ndarray:
        if cls.SCALE == "lin":
            return np.linspace(cls.MIN, cls.MAX, cls.COUNT, True)
        if cls.SCALE == "geom":
            return np.geomspace(cls.MIN, cls.MAX, cls.COUNT, True)
        if cls.SCALE == "log":
            return np.logspace(cls.MIN, cls.MAX, cls.COUNT, True)

        raise ValueError()

    @classmethod
    @property
    def axis_scale(cls) -> str:
        if cls.SCALE == "lin":
            return "linear"
        if cls.SCALE == "geom":
            return "log"
        if cls.SCALE == "log":
            return "log"

        raise ValueError()


class ParticleSizeRatioStudy(DimlessParameterStudy):
    KEY = "particle_size_ratio"
    TITLE = r"Particle Size Ratio $\Radius_2 / \Radius_1$"
    MIN = 1
    MAX = 10
    SCALE = "lin"
    COUNT = 10

    @property
    def input(self) -> Input:
        model = super().input
        model.particle2.radius *= self.real_value
        model.particle2.x = model.particle1.radius + model.particle2.radius - INTRUSION
        model.particle2.node_count = int(model.particle2.node_count * self.real_value)
        return model


ParticleSizeRatioStudy.INSTANCES = [ParticleSizeRatioStudy(value=v) for v in ParticleSizeRatioStudy.values]


class SurfaceBoundaryEnergyStudy(DimlessParameterStudy):
    KEY = "surface_boundary_energy"
    TITLE = r"Interface Energy Ratio $\InterfaceEnergy_{\GrainBoundary} / \InterfaceEnergy_{\Surface}$"
    MIN = 0.1
    MAX = 1.9
    SCALE = "lin"
    COUNT = 19

    @property
    def input(self) -> Input:
        model = super().input
        model.material1.grain_boundary.energy = model.material1.surface.energy * self.real_value / 2
        model.material2.grain_boundary.energy = model.material2.surface.energy * self.real_value / 2
        return model


SurfaceBoundaryEnergyStudy.INSTANCES = [SurfaceBoundaryEnergyStudy(value=v) for v in SurfaceBoundaryEnergyStudy.values]


class SurfaceBoundaryDiffusionStudy(DimlessParameterStudy):
    KEY = "surface_boundary_diffusion"
    TITLE = r"Diffusion Coefficient Ratio $\DiffusionCoefficient_{\GrainBoundary} / \DiffusionCoefficient_{\Surface}$"
    MIN = 0.01
    MAX = 10
    SCALE = "geom"
    COUNT = 16

    @property
    def input(self) -> Input:
        model = super().input
        model.material1.grain_boundary.diffusion_coefficient = (
            model.material1.surface.diffusion_coefficient * self.real_value / 2
        )
        model.material2.grain_boundary.diffusion_coefficient = (
            model.material2.surface.diffusion_coefficient * self.real_value / 2
        )
        return model


SurfaceBoundaryDiffusionStudy.INSTANCES = [
    SurfaceBoundaryDiffusionStudy(value=v) for v in SurfaceBoundaryDiffusionStudy.values
]


class DiffusionAsymmetricStudy(DimlessParameterStudy):
    KEY = "diffusion_asymmetric"
    TITLE = r"Diffusion Coefficient Ratio $\DiffusionCoefficient_{2} / \DiffusionCoefficient_{1}$"
    MIN = 1
    MAX = 100
    SCALE = "geom"

    @property
    def input(self) -> Input:
        model = super().input
        model.material2.surface.diffusion_coefficient = model.material1.surface.diffusion_coefficient * self.real_value
        model.material2.grain_boundary.diffusion_coefficient = (
            model.material1.grain_boundary.diffusion_coefficient * self.real_value
        )
        return model


DiffusionAsymmetricStudy.INSTANCES = [DiffusionAsymmetricStudy(value=v) for v in DiffusionAsymmetricStudy.values]


class SurfaceEnergyAsymmetricStudy(DimlessParameterStudy):
    KEY = "surface_energy_asymmetric"
    TITLE = r"Surface Energy Ratio $\InterfaceEnergy_{\Surface2} / \InterfaceEnergy_{\Surface1}$"
    MIN = 1
    MAX = 3
    SCALE = "lin"
    COUNT = 9

    @property
    def input(self) -> Input:
        model = super().input
        model.duration *= 1e1
        model.material2.surface.energy = model.material1.surface.energy * self.real_value
        return model


SurfaceEnergyAsymmetricStudy.INSTANCES = [
    SurfaceEnergyAsymmetricStudy(value=v) for v in SurfaceEnergyAsymmetricStudy.values
]


class OvalityTipTipStudy(DimlessParameterStudy):
    KEY = "ovality_tip_tip"
    TITLE = r"Ovality $\Ovality$"
    MIN = 1.0
    MAX = 3.0
    COUNT = 5
    SCALE = "lin"

    @property
    def input(self) -> Input:
        model = super().input
        model.particle1.ovality = self.real_value
        model.particle2.ovality = self.real_value
        model.particle2.x = (
            model.particle1.radius * np.sqrt(model.particle1.ovality)
            + model.particle2.radius * np.sqrt(model.particle2.ovality)
            - INTRUSION
        )
        return model


OvalityTipTipStudy.INSTANCES = [OvalityTipTipStudy(value=v) for v in OvalityTipTipStudy.values]


class OvalityTipFlankStudy(DimlessParameterStudy):
    KEY = "ovality_tip_flank"
    TITLE = r"Ovality $\Ovality$"
    MIN = 1.0
    MAX = 3.0
    COUNT = 5
    SCALE = "lin"

    @property
    def input(self) -> Input:
        model = super().input
        model.particle1.ovality = self.real_value
        model.particle2.ovality = self.real_value
        model.particle2.rotation_angle = np.pi / 2
        model.particle2.x = (
            model.particle1.radius * np.sqrt(model.particle1.ovality)
            + model.particle2.radius / np.sqrt(model.particle2.ovality)
            - INTRUSION
        )
        return model


OvalityTipFlankStudy.INSTANCES = [OvalityTipFlankStudy(value=v) for v in OvalityTipFlankStudy.values]


class OvalityFlankFlankStudy(DimlessParameterStudy):
    KEY = "ovality_flank_flank"
    TITLE = r"Ovality $\Ovality$"
    MIN = 1.0
    MAX = 3.0
    COUNT = 5
    SCALE = "lin"

    @property
    def input(self) -> Input:
        model = super().input
        model.particle1.ovality = self.real_value
        model.particle2.ovality = self.real_value
        model.particle1.rotation_angle = np.pi / 2
        model.particle2.rotation_angle = np.pi / 2
        model.particle2.x = (
            model.particle1.radius / np.sqrt(model.particle1.ovality)
            + model.particle2.radius / np.sqrt(model.particle2.ovality)
            - INTRUSION
        )
        return model


OvalityFlankFlankStudy.INSTANCES = [OvalityFlankFlankStudy(value=v) for v in OvalityFlankFlankStudy.values]

STUDIES: list[type[StudyBase]] = [
    TimeStepStudy,
    SurfaceRemeshingStudy,
    NeckRemeshingStudy,
    LongRunStudy,
    ParticleSizeRatioStudy,
    SurfaceBoundaryEnergyStudy,
    SurfaceBoundaryDiffusionStudy,
    SurfaceEnergyAsymmetricStudy,
    DiffusionAsymmetricStudy,
    OvalityTipTipStudy,
    OvalityTipFlankStudy,
    OvalityFlankFlankStudy,
]
