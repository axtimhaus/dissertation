import re
from pathlib import Path

import tomlkit

from dissertation.config import JINJA_ENV, ROOT_DIR, in_build_dir

RE_ARGUMENT = re.compile(r"#\d")


def create_command_def(name: str, code: str):
    if "#" in code:
        arguments = RE_ARGUMENT.findall(code)
        return rf"\gdef\{name}{''.join(arguments)}{{{code}}}"
    else:
        return rf"\gdef\{name}{{{code}}}"


def task_symbols(
    toml_file=ROOT_DIR / "symbols.toml",
    produces=ROOT_DIR / "symbols.sty",
):
    input_text = toml_file.read_text()

    data = tomlkit.loads(input_text)

    lines = [create_command_def(n, c) for n, c in data.items()]

    produces.write_text("\n".join(lines))


def task_list_of_symbols(
    toml_file=Path("symbols.toml"),
    template=ROOT_DIR / "list_of_symbols.tex",
    produces=in_build_dir(ROOT_DIR / "list_of_symbols.tex"),
):
    input_text = toml_file.read_text()
    data = tomlkit.loads(input_text)
    template = JINJA_ENV.get_template(template.relative_to(ROOT_DIR).as_posix())
    rendered = template.render(
        symbols=[
            dict(key=k, value=v.replace("#1", "\\bullet"), comment=c.strip(" #"))
            for k, v in data.items()
            if (c := data.item(k).trivia.comment)
        ]
    )
    produces.write_text(rendered)
