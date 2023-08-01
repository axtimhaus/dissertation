import pytask
import pytask_latex.compilation_steps as cs

from dissertation.config import ROOT_DIR


# noinspection PyTypeChecker
@pytask.mark.latex(
    script="dissertation.tex",
    document="dissertation.pdf",
    compilation_steps=cs.latexmk(
        options=("-r", ROOT_DIR / "latexmkrc")
    )
)
@pytask.mark.depends_on("symbols.sty")
def task_build_latex():
    pass
