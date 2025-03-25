import subprocess
from pathlib import Path
from uuid import UUID

import numpy as np
from rich.markup import escape

from dissertation.sim.two_particle.input import (
    Input,
    InterfaceInput,
    MaterialInput,
    ParticleInput,
)

THIS_DIR = Path(__file__).parent
RUN_DIR = THIS_DIR / "run"

PARTICLE1_ID = UUID("989b9875-2a6b-40c3-ab2f-5ebc96682dbe")
PARTICLE2_ID = UUID("10cac1cc-6205-4b91-85b4-4e9d6f126274")

PARTICLE1 = ParticleInput(id=PARTICLE1_ID, radius=100e-6)
PARTICLE2 = PARTICLE1.model_copy(
    deep=True,
    update={
        "id": PARTICLE2_ID,
        "x": 1.99 * PARTICLE1.radius,
        "rotation_angle": np.pi,
    },
)

SURFACE = InterfaceInput(energy=0.9, diffusion_coefficient=1.65e-10)
GRAIN_BOUNDARY = InterfaceInput(
    energy=SURFACE.energy * 0.5,
    diffusion_coefficient=SURFACE.diffusion_coefficient,
)

MATERIAL = MaterialInput(
    surface=SURFACE.model_copy(),
    density=1.8e3,
    molar_mass=101.96e-3,
)

INPUT = Input(
    particle1=PARTICLE1,
    particle2=PARTICLE2,
    material1=MATERIAL,
    material2=MATERIAL,
    grain_boundary=GRAIN_BOUNDARY,
    gas_constant=8.31446261815324,
    temperature=1273,
    vacancy_concentration=1e-4,
    duration=3.6e5,
)


def task_create_input(
    produces=RUN_DIR / "input.json",
):
    produces.parent.mkdir(exist_ok=True, parents=True)
    produces.write_text(INPUT.model_dump_json(indent=4))


def task_run(
    input_file=RUN_DIR / "input.json",
    produces=RUN_DIR / "output.parquet",
    csharp_proj=THIS_DIR / "two_particle.csproj",
):
    result = subprocess.run(
        [
            "dotnet",
            "run",
            "--project",
            str(csharp_proj),
            str(input_file),
            str(produces),
        ],
        cwd=str(input_file.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    print("=== OUT ===\n", escape(result.stdout))
    print("=== ERR ===\n", escape(result.stderr))

    result.check_returncode()
