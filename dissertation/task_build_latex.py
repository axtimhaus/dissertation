from pathlib import Path

import pytask
import pytask_latex.compilation_steps as cs

from dissertation.config import ROOT_DIR


# noinspection PyTypeChecker
@pytask.mark.latex(
    script=Path("dissertation.tex"),
    document=Path("dissertation.pdf"),
    compilation_steps=cs.latexmk(
        options=[
            "-r",
            str(ROOT_DIR / "latexmkrc"),
            "-g",
        ],
    ),
)
def task_build_latex(
    symbols_file=Path("symbols.sty"),
):
    pass
