from pathlib import Path

import pytask
import tomlkit

PREAMBLE = r"""
\documentclass{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
"""


@pytask.mark.depends_on("symbols.sty")
@pytask.mark.produces("textext_preamble.tex")
def task_textext_preamble(depends_on: Path, produces: Path):
    symbols_text = depends_on.read_text()

    produces.write_text(PREAMBLE + symbols_text)
