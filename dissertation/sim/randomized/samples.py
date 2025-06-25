import itertools
from uuid import uuid5
from scipy.stats import norm
import numpy as np
from pathlib import Path

from dissertation.config import in_build_dir
from dissertation.sim.randomized.input import (
    REFERENCE_MATERIAL,
    Input,
    ParticleInput,
    MaterialInput,
    InterfaceInput,
    NAMESPACE,
)

THIS_DIR = Path(__file__).parent
SEED = np.random.default_rng(78358734567934)
SAMPLE_COUNT = 10
PARTICLE_COUNT = 3

PARTICLE_SIZE = norm(loc=100e-6, scale=30e-6)
INITIAL_DISTANCE = 400e-6
COORDS = [(0,0), (INITIAL_DISTANCE, 0), (INITIAL_DISTANCE /2, INITIAL_DISTANCE / 2 * np.sqrt(3))]


def create_input(sample: int):
    particles = [
        ParticleInput(
            id=uuid5(NAMESPACE, f"{sample}/{i}"),
            x=x,
            y=y,
            radius=PARTICLE_SIZE.rvs(1, random_state=SEED),
            material=REFERENCE_MATERIAL,
            grain_boundaries={
                uuid5(NAMESPACE, f"{sample}/{j}"): REFERENCE_MATERIAL.surface.model_copy(update=dict(energy=0.45))
                for j in range(PARTICLE_COUNT)
                if not j == i
            },
        )
        for i, (x, y) in zip(range(PARTICLE_COUNT), COORDS, strict=True)
    ]

    return Input(particles=particles)


SAMPLES = [create_input(i) for i in range(SAMPLE_COUNT)]


def dir(i):
    return in_build_dir(THIS_DIR / "samples" / f"{i}")
