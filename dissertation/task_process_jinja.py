import pytask
from pathlib import Path
import tomlkit

from dissertation.config import ROOT_DIR, BUILD_DIR, JINJA_ENV

SYMBOL_INDEX = ROOT_DIR / "symbol_index.tex"


@pytask.mark.task(id=str(SYMBOL_INDEX.relative_to(ROOT_DIR)))
@pytask.mark.depends_on({
    "template": SYMBOL_INDEX,
    "data": "symbols.toml"
})
@pytask.mark.produces(BUILD_DIR / SYMBOL_INDEX.relative_to(ROOT_DIR))
@pytask.mark.skip
def task_process_symbols_index(depends_on: dict[str, Path], produces: Path):
    template = JINJA_ENV.get_template(str(depends_on["template"].relative_to(ROOT_DIR)))

    symbols = tomlkit.loads(depends_on["data"].read_text())["symbol"]

    result = template.render(symbols=symbols)

    produces.write_text(result)
