from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from uuid import uuid5

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import Mixture, Uniform, beta, make_distribution, weibull_min

from dissertation.config import in_build_dir
from dissertation.sim.randomized.input import (
    NAMESPACE,
    REFERENCE_GRAIN_BOUNDARY,
    REFERENCE_MATERIAL,
    REFERENCE_PARTICLE,
    Input,
    ParticleInput,
)

THIS_DIR = Path(__file__).parent
SAMPLE_COUNT = 500
PARTICLE_COUNT = 3
NODE_COUNT = 200
DISCRETIZATION_WIDTH = 2 * np.pi * REFERENCE_PARTICLE.radius / NODE_COUNT

_Weibull = make_distribution(weibull_min)
Beta = make_distribution(beta)
ROTATION_DIST = Uniform(a=0, b=2 * np.pi)


def Weibull(c, s, loc=0):
    return s * _Weibull(c=c) + loc


def Weibull2(c1, s1, c2, s2, w, l1=0, l2=0):
    return Mixture([Weibull(c1, s1, l1), Weibull(c2, s2, l2)], weights=[w, 1 - w])


def particle_coords(distance: float):
    distance *= 1.1
    return [(0, 0), (distance, 0), (distance / 2, distance / 2 * np.sqrt(3))]


@dataclass
class Categorical:
    values: np.ndarray
    probabilities: np.ndarray

    def sample(self, shape, rng):
        return rng.choice(self.values, size=shape, p=self.probabilities)


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    display: str
    samples: list[Input]

    LINE_STYLE: ClassVar[dict]

    def dir(self, sample: int | None = None) -> Path:
        path = in_build_dir(THIS_DIR / "cases") / self.key
        return path if sample is None else path / str(sample)


class NominalCase(Case):
    LINE_STYLE = dict(color="C0")

    @classmethod
    def create_input(cls, sample: int):
        particles = [
            ParticleInput(
                id=uuid5(NAMESPACE, f"{sample}/{i}"),
                x=x,
                y=y,
                radius=REFERENCE_PARTICLE.radius,
                material=REFERENCE_MATERIAL,
                grain_boundaries={
                    uuid5(NAMESPACE, f"{sample}/{j}"): REFERENCE_GRAIN_BOUNDARY for j in range(PARTICLE_COUNT) if j != i
                },
                node_count=NODE_COUNT,
            )
            for i, (x, y) in zip(range(PARTICLE_COUNT), particle_coords(1.1 * REFERENCE_PARTICLE.radius), strict=True)
        ]

        return Input(particles=particles)


class CircularCase(Case):
    PARTICLE_SIZE_DIST: ClassVar = Weibull2(1.819, 1281.114, 6.858, 1671.525, 0.413)
    RNG: ClassVar = np.random.default_rng(42)

    LINE_STYLE = dict(color="C0")

    @classmethod
    def create_input(cls, sample: int):
        radii = cls.PARTICLE_SIZE_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG) / 1e6
        distance = 2 * np.max(radii)

        particles = [
            ParticleInput(
                id=uuid5(NAMESPACE, f"{sample}/{i}"),
                x=x,
                y=y,
                radius=radii[i],
                material=REFERENCE_MATERIAL,
                grain_boundaries={
                    uuid5(NAMESPACE, f"{sample}/{j}"): REFERENCE_GRAIN_BOUNDARY for j in range(PARTICLE_COUNT) if j != i
                },
                node_count=np.ceil(2 * np.pi * radii[i] / DISCRETIZATION_WIDTH),
            )
            for i, (x, y) in zip(range(PARTICLE_COUNT), particle_coords(distance), strict=True)
        ]

        return Input(particles=particles)


class OvalCase(Case):
    PARTICLE_SIZE_DIST: ClassVar = Weibull2(5.227, 1641.123, 5.193, 568.826, 0.854)
    OVALITY_DIST: ClassVar = Weibull(1.325, 0.377, 1)
    RNG: ClassVar = np.random.default_rng(42)

    LINE_STYLE = dict(color="C0")

    @classmethod
    def create_input(cls, sample: int):
        radii = cls.PARTICLE_SIZE_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG) / 1e6
        ovalities = cls.OVALITY_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        rotations = ROTATION_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        distance = 2 * np.max(radii * ovalities)

        particles = [
            ParticleInput(
                id=uuid5(NAMESPACE, f"{sample}/{i}"),
                x=x,
                y=y,
                radius=radii[i],
                ovality=ovalities[i],
                rotation_angle=rotations[i],
                material=REFERENCE_MATERIAL,
                grain_boundaries={
                    uuid5(NAMESPACE, f"{sample}/{j}"): REFERENCE_GRAIN_BOUNDARY for j in range(PARTICLE_COUNT) if j != i
                },
                node_count=np.ceil(2 * np.pi * radii[i] / DISCRETIZATION_WIDTH),
            )
            for i, (x, y) in zip(range(PARTICLE_COUNT), particle_coords(distance), strict=True)
        ]

        return Input(particles=particles)


class ShapeCase(Case):
    PARTICLE_SIZE_DIST: ClassVar = Weibull2(5.199, 1630.037, 5.133, 566.077, 0.854)
    OVALITY_DIST: ClassVar = Weibull(1.305, 0.378, 1)
    HEIGHT_DIST: ClassVar = Beta(a=3.938, b=43.030)
    SHIFT_DIST: ClassVar = Uniform(a=0, b=0.5)
    COUNT_DIST: ClassVar = Categorical(np.arange(3, 8 + 1), np.asarray([0.680, 0.179, 0.091, 0.032, 0.011, 0.007]))
    RNG: ClassVar = np.random.default_rng(42)

    LINE_STYLE = dict(color="C0")

    @classmethod
    def create_input(cls, sample: int):
        radii = cls.PARTICLE_SIZE_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG) / 1e6
        ovalities = cls.OVALITY_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        rotations = ROTATION_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        heights = cls.HEIGHT_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        shifts = cls.SHIFT_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        counts = cls.COUNT_DIST.sample((PARTICLE_COUNT,), rng=cls.RNG)
        distance = 2 * np.max(radii * ovalities * (1 + heights))

        particles = [
            ParticleInput(
                id=uuid5(NAMESPACE, f"{sample}/{i}"),
                x=x,
                y=y,
                radius=radii[i],
                ovality=ovalities[i],
                peak_count=counts[i],
                peak_height=heights[i],
                peak_shift=shifts[i],
                rotation_angle=rotations[i],
                material=REFERENCE_MATERIAL,
                grain_boundaries={
                    uuid5(NAMESPACE, f"{sample}/{j}"): REFERENCE_GRAIN_BOUNDARY for j in range(PARTICLE_COUNT) if j != i
                },
                node_count=np.ceil(2 * np.pi * radii[i] / DISCRETIZATION_WIDTH),
            )
            for i, (x, y) in zip(range(PARTICLE_COUNT), particle_coords(distance), strict=True)
        ]

        return Input(particles=particles)


CASES = [
    CircularCase(
        key="circular",
        display="Circular",
        samples=[CircularCase.create_input(i) for i in range(SAMPLE_COUNT)],
    ),
    OvalCase(
        key="oval",
        display="Oval",
        samples=[OvalCase.create_input(i) for i in range(SAMPLE_COUNT)],
    ),
    ShapeCase(
        key="shape",
        display="Shape",
        samples=[ShapeCase.create_input(i) for i in range(SAMPLE_COUNT)],
    ),
]
