import subprocess
import time
from pathlib import Path

from pytask import task, mark

from dissertation.sim.two_particle.studies import STUDIES
from rich.markup import escape

THIS_DIR = Path(__file__).parent

for t in STUDIES:
    for study in t.INSTANCES:

        @task(id=study.key)
        @mark.sim
        def task_create_input(
            study=study,
            produces=study.dir / "input.json",
            studies_module=THIS_DIR / "studies.py",
            input_module=THIS_DIR / "input.py",
        ):
            produces.parent.mkdir(exist_ok=True, parents=True)
            produces.write_text(study.input.model_dump_json(indent=4))

        @task(id=study.key)
        @mark.persis
        @mark.sim
        def task_run(
            input_file=study.dir / "input.json",
            produces={"output": study.dir / "output.parquet", "time": study.dir / "time.txt"},
            csharp_proj=THIS_DIR / "two_particle.csproj",
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
