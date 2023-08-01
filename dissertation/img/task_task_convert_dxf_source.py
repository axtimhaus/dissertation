import subprocess

import pytask
from pathlib import Path

THIS_DIR = Path(__file__).parent
FILES = THIS_DIR.rglob("__src__/*.dxf")

for f in FILES:

    f_pdf = f.with_suffix(".pdf")


    @pytask.mark.task(id=str(f.relative_to(THIS_DIR)))
    @pytask.mark.skipif(not f_pdf.exists(), reason="DXF source was not exported to PDF")
    @pytask.mark.depends_on(f)
    @pytask.mark.produces([f.parent.parent / (f.stem + "." + e) for e in ["png", "svg", "pdf"]])
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
