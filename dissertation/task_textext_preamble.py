from pathlib import Path

import pytask

PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math-style=ISO]{unicode-math}
\usepackage{siunitx}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setmathfont{TeX Gyre Termes Math}
"""


def task_textext_preamble(
        sty_file=Path("symbols.sty"),
        produces=Path("textext_preamble.tex")
):
    symbols_text = sty_file.read_text()
    preamble = PREAMBLE + symbols_text

    produces.write_text(preamble)
