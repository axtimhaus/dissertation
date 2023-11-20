from pathlib import Path
import matplotlib as mpl

ROOT_DIR = Path(__file__).parent
BUILD_DIR = ROOT_DIR / ".build"

BUILD_DIR.mkdir(exist_ok=True)


def in_build_dir(f: Path):
    return BUILD_DIR / f.relative_to(ROOT_DIR)


IMAGE_FILE_FORMATS = [
    "pdf",
    "png",
]

IMAGE_FILE_SUFFIXES = ["." + f for f in IMAGE_FILE_FORMATS]

mpl.use('pgf')

PREAMBLE_FILE = Path(ROOT_DIR / "textext_preamble.tex")

if PREAMBLE_FILE.exists():
    preamble = PREAMBLE_FILE.read_text()
    mpl.rcParams.update({
        "pgf.preamble": preamble,
        "text.latex.preamble": preamble
    })

mpl.rcParams.update({
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False,
    "figure.autolayout": True,
})

import jinja2

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))
