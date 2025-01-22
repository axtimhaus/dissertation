import subprocess
from pathlib import Path

import pytask

from dissertation.config import IMAGE_FILE_SUFFIXES, in_build_dir

THIS_DIR = Path(__file__).parent
FILES = THIS_DIR.rglob("*.svg")

for f in FILES:

    @pytask.task(id=str(f.relative_to(THIS_DIR)))
    def task_task_convert_svg_source(
        source: Path = f, produces: list[Path] = [in_build_dir(f.with_suffix(e)) for e in IMAGE_FILE_SUFFIXES]
    ):
        for t in produces:
            result = subprocess.run(
                ["inkscape", "-D", "-d", "600", "-o", str(t), str(source)], capture_output=True, text=True, check=False
            )

            print(result.stdout)
            result.check_returncode()
