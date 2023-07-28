import os
from pathlib import Path

import pytask
import tomlkit


def create_command_def(name: str, code: str):
    if "#" in code:
        par_count = code.count("#")
        return rf"\newcommand{{\{name}}}[{par_count}]{{{{{code}}}}}"
    else:
        return rf"\newcommand{{\{name}}}{{{{{code}}}}}"


@pytask.mark.depends_on("symbols.toml")
@pytask.mark.produces("symbols.sty")
def task_symbols(depends_on: Path, produces: Path):
    input_text = depends_on.read_text()

    data = tomlkit.loads(input_text)

    lines = [
        create_command_def(n, c)
        for n, c in data.items()
    ]

    produces.write_text("\n".join(lines))
