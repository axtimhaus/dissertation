import pytask


@pytask.mark.latex(
    script="dissertation.tex",
    document="dissertation.pdf",
)
def task_build_latex():
    pass
