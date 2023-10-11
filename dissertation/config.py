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
    "svg"
]

IMAGE_FILE_SUFFIXES = ["." + f for f in IMAGE_FILE_FORMATS]

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": rf"\usepackage{{{ROOT_DIR / 'symbols'}}}\usepackage{{siunitx}}\usepackage{{amsmath}}",
    "font.family": "serif",
    "figure.autolayout": True,
})

import jinja2

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))
