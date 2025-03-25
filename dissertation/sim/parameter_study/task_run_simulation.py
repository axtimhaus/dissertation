import subprocess
from pathlib import Path

from pytask import task, mark

from dissertation.sim.parameter_study.studies import STUDIES
from rich.markup import escape

THIS_DIR = Path(__file__).parent

for study in STUDIES:
    for value in study.parameter_values:

        @task(id=f"{study}/{value}")
        @mark.parameter_study
        def task_create_input(
            study=study,
            value=value,
            produces=study.dir(value) / "input.json",
            studies_module=THIS_DIR / "studies.py",
        ):
            produces.parent.mkdir(exist_ok=True, parents=True)
            produces.write_text(study.input_for(value).model_dump_json(indent=4))

        @task(id=f"{study}/{value}")
        @mark.persist
        @mark.sim
        @mark.parameter_study
        def task_run(
            input_file=study.dir(value) / "input.json",
            produces=study.dir(value) / "output.parquet",
            csharp_proj=THIS_DIR.parent / "two_particle" / "two_particle.csproj",
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
