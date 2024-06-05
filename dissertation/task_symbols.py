from pathlib import Path

import tomlkit


def create_command_def(name: str, code: str):
    if "#" in code:
        par_count = code.count("#")
        return rf"\newcommand{{\{name}}}[{par_count}]{{{{{code}}}}}"
    else:
        return rf"\newcommand{{\{name}}}{{{{{code}}}}}"


def task_symbols(
    toml_file=Path("symbols.toml"),
    produces=Path("symbols.sty"),
):
    input_text = toml_file.read_text()

    data = tomlkit.loads(input_text)

    lines = [create_command_def(n, c) for n, c in data.items()]

    produces.write_text("\n".join(lines))
