import subprocess

import pytask
from pathlib import Path

THIS_DIR = Path(__file__).parent
FILES = THIS_DIR.rglob("__src__/*.svg")

for f in FILES:

    @pytask.mark.task(id=str(f.relative_to(THIS_DIR)))
    @pytask.mark.depends_on(f)
    @pytask.mark.produces([f.parent.parent / (f.stem + "." + e) for e in ["png", "svg", "pdf"]])
    def task_task_convert_svg_source(depends_on: Path, produces: dict[..., Path]):
        for p in produces.values():
            result = subprocess.run([
                "inkscape",
                "-D",
                "-d", "600",
                "-o", str(p),
                str(depends_on)
            ], capture_output=True, text=True)

            print(result.stdout)
            result.check_returncode()
