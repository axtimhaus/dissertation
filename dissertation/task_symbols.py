from pathlib import Path

import pytask
import toml

from dissertation.config import JINJA_ENV, ROOT_DIR


@pytask.mark.depends_on({
    "data": "symbols.toml",
    "template": "_symbols.sty"
})
@pytask.mark.produces("symbols.sty")
def task_symbols(depends_on: dict[str, Path], produces: Path):
    template = JINJA_ENV.get_template(str(depends_on["template"].relative_to(ROOT_DIR)))

    symbols = toml.loads(depends_on["data"].read_text())["symbol"]

    result = template.render(symbols=symbols)

    produces.write_text(result)