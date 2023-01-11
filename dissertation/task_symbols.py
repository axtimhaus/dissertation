from pathlib import Path

import pytask
import toml

from dissertation.config import JINJA_ENV, ROOT_DIR


@pytask.mark.depends_on("symbols.toml")
@pytask.mark.produces("symbols.sty")
def task_symbols(depends_on: Path, produces: Path):
    data = toml.loads(depends_on.read_text())

    lines = [
        rf"\newcommand{{\{s['name']}}}{{{{{s['code']}}}}}"
        for s in data["symbol"]
    ]

    produces.write_text("\n".join(lines))
