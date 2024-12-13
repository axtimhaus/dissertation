import subprocess
from pathlib import Path

from pytask import task

from dissertation.sim.parameter_study.studies import STUDIES

THIS_DIR = Path(__file__).parent

for study in STUDIES:
    for value in study.parameter_values:

        @task(id=f"{study}/{value}")
        def task_create_input(
            study=study,
            value=value,
            produces=study.dir(value) / "input.json",
            studies_module=THIS_DIR / "studies.py",
        ):
            produces.parent.mkdir(exist_ok=True, parents=True)
            produces.write_text(study.input_for(value).model_dump_json(indent=4))

        @task(id=f"{study}/{value}")
        def task_run(
            input_file=study.dir(value) / "input.json",
            produces=study.dir(value) / "output.json",
            csharp_proj=THIS_DIR / "parameter_study" / "parameter_study.csproj",
            csharp_program=THIS_DIR / "parameter_study" / "Program.cs",
        ):
            subprocess.run(
                ["dotnet", "run", str(input_file), str(produces), "--project", str(csharp_proj)],
                cwd=str(input_file.parent),
                check=False,
            )
