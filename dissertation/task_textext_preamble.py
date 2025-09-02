from pathlib import Path

import matplotlib as mpl
import pytask

PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math-style=ISO]{unicode-math}
\usepackage[mode=match]{siunitx}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setmathfont{TeX Gyre Termes Math}
"""


@pytask.mark.try_first
def task_textext_preamble(sty_file=Path("symbols.sty"), produces=Path("textext_preamble.tex")):
    symbols_text = sty_file.read_text()
    preamble = PREAMBLE + symbols_text

    produces.write_text(preamble)

    mpl.rcParams.update({"pgf.preamble": preamble, "text.latex.preamble": preamble})
