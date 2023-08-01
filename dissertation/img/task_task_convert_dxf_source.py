import subprocess

import pytask
from pathlib import Path

from dissertation.config import in_build_dir, IMAGE_FILE_SUFFIXES

THIS_DIR = Path(__file__).parent
FILES = THIS_DIR.rglob("*.dxf")

for f in FILES:

    f_pdf = f.with_suffix(".pdf")


    @pytask.mark.task(id=str(f.relative_to(THIS_DIR)))
    @pytask.mark.skipif(not f_pdf.exists(), reason="DXF source was not exported to PDF")
    @pytask.mark.depends_on(f)
    @pytask.mark.produces([in_build_dir(f.with_suffix(e)) for e in IMAGE_FILE_SUFFIXES])
    def task_task_convert_dxf_source(depends_on: Path, produces: dict[..., Path]):
        for p in produces.values():
            result = subprocess.run([
                "inkscape",
                "-D",
                "-d", "600",
                "-o", str(p),
                str(depends_on.with_suffix(".pdf"))
            ], capture_output=True, text=True)

            print(result.stdout)
            result.check_returncode()
