import subprocess
from pathlib import Path

from _pytask.nodes import PythonNode
from pytask import task
from typing_extensions import Annotated

from dissertation.sim.parameter_study.data_model import ParameterStudy
from dissertation.sim.parameter_study.studies import STUDIES, hash

THIS_DIR = Path(__file__).parent

for study in STUDIES:
    data_dir = THIS_DIR / "runs" / str(study)
    data_dir.mkdir(exist_ok=True, parents=True)

    @task()
    def task_create_input(
        study: Annotated[ParameterStudy, PythonNode(value=study, hash=hash)],
        produces = data_dir / "input.json"
    ):
        produces.write_text(study.model_dump_json(indent=4))


    @task()
    def task_run_simulation(
        input_file = data_dir / "input.json",
        produces = data_dir / "output.json",
        csharp_proj= THIS_DIR / "parameter_study" / "parameter_study.csproj",
        csharp_program = THIS_DIR / "parameter_study" / "Program.cs",
    ):
        subprocess.run(
            [
                "dotnet", "run",
                str(input_file),
                str(produces),
                "--project", str(csharp_proj)
            ],
            cwd=str(data_dir), check=False
        )
