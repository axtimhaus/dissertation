from pathlib import Path

import pytask

PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math-style=ISO]{unicode-math}
\usepackage{siunitx}
"""


@pytask.mark.depends_on("symbols.sty")
@pytask.mark.produces("textext_preamble.tex")
def task_textext_preamble(depends_on: Path, produces: Path):
    symbols_text = depends_on.read_text()
    preamble = PREAMBLE + symbols_text

    produces.write_text(preamble)
