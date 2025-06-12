import subprocess
import time
from pathlib import Path

from pytask import mark, task
from rich.markup import escape

from dissertation.sim.packings.cases import CASES

THIS_DIR = Path(__file__).parent

for case in CASES:
        @task(id=case.key)
        @mark.sim
        def task_packings_create_input(
            case=case,
            produces=case.dir / "input.json",
            cases_module=THIS_DIR / "cases.py",
            input_module=THIS_DIR / "input.py",
        ):
            produces.parent.mkdir(exist_ok=True, parents=True)
            produces.write_text(case.input.model_dump_json(indent=4))

        @task(id=case.key)
        @mark.persist
        @mark.sim
        def task_packings_run(
            input_file=case.dir / "input.json",
            produces={"output": case.dir / "output.parquet", "time": case.dir / "time.txt"},
            csharp_proj=THIS_DIR / "packings.csproj",
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
