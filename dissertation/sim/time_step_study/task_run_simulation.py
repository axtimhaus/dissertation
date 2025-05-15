import subprocess
import time
from pathlib import Path

from pytask import task, mark

from dissertation.sim.time_step_study.studies import STUDIES
from rich.markup import escape

THIS_DIR = Path(__file__).parent

for study in STUDIES:

    @task(id=f"{study}")
    @mark.persist
    @mark.time_step_study
    def task_create_input(
        study=study,
        produces=study.dir() / "input.json",
        # studies_module=THIS_DIR / "studies.py",
    ):
        produces.parent.mkdir(exist_ok=True, parents=True)
        produces.write_text(study.input.model_dump_json(indent=4))

    @task(id=f"{study}")
    @mark.persist
    @mark.sim
    @mark.time_step_study
    def task_run(
        input_file=study.dir() / "input.json",
        produces={"output": study.dir() / "output.parquet", "time": study.dir() / "time.txt"},
        csharp_proj=THIS_DIR.parent / "two_particle" / "two_particle.csproj",
    ):
        start = time.time()
        result = subprocess.run(
            [
                "dotnet",
                "run",
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
