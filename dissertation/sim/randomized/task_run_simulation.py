import subprocess
import time
from pathlib import Path

from pytask import mark, task
from rich.markup import escape

from dissertation.sim.randomized.cases import CASES, NOMINAL, NOMINAL_MODIFIED_AVERAGE

THIS_DIR = Path(__file__).parent

for case in CASES:
    for i, sample in enumerate(case.samples):

        @task(id=f"{case.key}/{i}")
        @mark.sim
        def task_randomized_create_sample(
            produces=case.dir(i) / "input.json",
            sample=sample,
            case_module=THIS_DIR / "cases.py",
            input_module=THIS_DIR / "input.py",
        ):
            produces.parent.mkdir(exist_ok=True, parents=True)
            produces.write_text(sample.model_dump_json(indent=4))

        @task(id=f"{case.key}/{i}")
        @mark.persist
        @mark.sim
        def task_randomized_run_sample(
            input_file=case.dir(i) / "input.json",
            produces={"output": case.dir(i) / "output.parquet", "time": case.dir(i) / "time.txt"},
            csharp_proj=THIS_DIR / "randomized.csproj",
        ):
            start = time.time()
            result = subprocess.run(
                [
                    "dotnet",
                    "run",
                    "-c",
                    "Release",
                    "--project",
                    str(csharp_proj),
                    str(input_file),
                    str(produces["output"]),
                ],
                cwd=str(input_file.parent),
                check=False,
                capture_output=True,
                text=True,
            )

            end = time.time()

            print("=== OUT ===\n", escape(result.stdout))
            print("=== ERR ===\n", escape(result.stderr))

            produces["time"].write_text(str(end - start))

            result.check_returncode()


@mark.sim
def task_randomized_create_nominal(
    produces=NOMINAL.dir() / "input.json",
    case_module=THIS_DIR / "cases.py",
    input_module=THIS_DIR / "input.py",
):
    produces.parent.mkdir(exist_ok=True, parents=True)
    produces.write_text(NOMINAL.input.model_dump_json(indent=4))


@mark.persist
@mark.sim
def task_randomized_run_nominal(
    input_file=NOMINAL.dir() / "input.json",
    produces={"output": NOMINAL.dir() / "output.parquet", "time": NOMINAL.dir() / "time.txt"},
    csharp_proj=THIS_DIR / "randomized.csproj",
):
    start = time.time()
    result = subprocess.run(
        [
            "dotnet",
            "run",
            "-c",
            "Release",
            "--project",
            str(csharp_proj),
            str(input_file),
            str(produces["output"]),
        ],
        cwd=str(input_file.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    end = time.time()

    print("=== OUT ===\n", escape(result.stdout))
    print("=== ERR ===\n", escape(result.stderr))

    produces["time"].write_text(str(end - start))

    result.check_returncode()


@mark.sim
def task_randomized_create_nominal_modified_average(
    produces=NOMINAL_MODIFIED_AVERAGE.dir() / "input.json",
    case_module=THIS_DIR / "cases.py",
    input_module=THIS_DIR / "input.py",
):
    produces.parent.mkdir(exist_ok=True, parents=True)
    produces.write_text(NOMINAL_MODIFIED_AVERAGE.input.model_dump_json(indent=4))


@mark.persist
@mark.sim
def task_randomized_run_nominal_modified_average(
    input_file=NOMINAL_MODIFIED_AVERAGE.dir() / "input.json",
    produces={
        "output": NOMINAL_MODIFIED_AVERAGE.dir() / "output.parquet",
        "time": NOMINAL_MODIFIED_AVERAGE.dir() / "time.txt",
    },
    csharp_proj=THIS_DIR / "randomized.csproj",
):
    start = time.time()
    result = subprocess.run(
        [
            "dotnet",
            "run",
            "-c",
            "Release",
            "--project",
            str(csharp_proj),
            str(input_file),
            str(produces["output"]),
        ],
        cwd=str(input_file.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    end = time.time()

    print("=== OUT ===\n", escape(result.stdout))
    print("=== ERR ===\n", escape(result.stderr))

    produces["time"].write_text(str(end - start))

    result.check_returncode()
