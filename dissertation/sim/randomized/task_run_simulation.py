import subprocess
import time
from pathlib import Path

from pytask import mark, task
from rich.markup import escape

from dissertation.sim.randomized.samples import SAMPLES, dir

THIS_DIR = Path(__file__).parent

for i, sample in enumerate(SAMPLES):

    @task(id=str(i))
    @mark.sim
    def task_randomized_create_sample(
        produces=dir(i) / "input.json",
        sample=sample,
        samples_module=THIS_DIR / "samples.py",
        input_module=THIS_DIR / "input.py",
    ):
        produces.parent.mkdir(exist_ok=True, parents=True)
        produces.write_text(sample.model_dump_json(indent=4))

    @task(id=str(i))
    @mark.persist
    @mark.sim
    def task_randomized_run(
        input_file=dir(i) / "input.json",
        produces={"output": dir(i) / "output.parquet", "time": dir(i) / "time.txt"},
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
