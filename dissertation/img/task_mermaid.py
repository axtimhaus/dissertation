import subprocess
import tempfile
from pathlib import Path

import pytask

from dissertation.config import IMAGE_FILE_SUFFIXES, in_build_dir

THIS_DIR = Path(__file__).parent
FILES = THIS_DIR.rglob("*.mmd")

for f in FILES:

    @pytask.task(id=str(f.relative_to(THIS_DIR)))
    def task_mermaid(
        source: Path = f, produces: list[Path] = [in_build_dir(f.with_suffix(e)) for e in IMAGE_FILE_SUFFIXES]
    ):
        tmp_pdf = tempfile.mktemp(".pdf")

        result = subprocess.run(
            [
                "mmdc",
                "-i",
                str(source),
                "-o",
                str(tmp_pdf),
                # "-e", "pdf",
                "--theme",
                "neutral",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        print(result.stdout)
        result.check_returncode()

        tmp_svg = tempfile.mktemp(".svg")

        result = subprocess.run(
            ["inkscape", "--export-id", "g1", "--export-id-only", "-l", "-o", str(tmp_svg), str(tmp_pdf)],
            capture_output=True,
            text=True,
            check=False,
        )

        for t in produces:
            result = subprocess.run(
                ["inkscape", "-D", "-d", "600", "-l", "-o", str(t), str(tmp_svg)],
                capture_output=True,
                text=True,
                check=False,
            )

            print(result.stdout)
            result.check_returncode()
