import pytask


@pytask.mark.latex(
    script="dissertation.tex",
    document="dissertation.pdf",
)
@pytask.mark.depends_on("symbols.sty")
def task_build_latex():
    pass
