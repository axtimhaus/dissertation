from pathlib import Path
from uuid import UUID

import jinja2
import matplotlib as mpl
import numpy as np

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


def image_produces(stem: Path):
    return [stem.with_suffix(s) for s in IMAGE_FILE_SUFFIXES]


mpl.use("pgf")

PREAMBLE_FILE = Path(ROOT_DIR / "textext_preamble.tex")

if PREAMBLE_FILE.exists():
    preamble = PREAMBLE_FILE.read_text()
    mpl.rcParams.update({"pgf.preamble": preamble, "text.latex.preamble": preamble})


def mm_to_inch(v):
    return np.asarray(v) / 25.4


DEFAULT_FIGSIZE = mm_to_inch([160, 70])
EVOLUTION_FIGSIZE = mm_to_inch([160, 80])

mpl.rcParams.update(
    {
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False,
        "figure.constrained_layout.use": True,
        "figure.dpi": 600,
        "figure.figsize": DEFAULT_FIGSIZE,
        "lines.linewidth": 1,
        "patch.linewidth": 1,
        "contour.linewidth": 1,
        "savefig.transparent": True,
    }
)


JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))

_integers = np.arange(1, 10)


def integer_log_space(factor_start: int, exp_start: int, factor_end: int, exp_end: int) -> np.typing.NDArray:
    locs = np.outer(10.0 ** np.arange(exp_start, exp_end + 1), _integers).reshape(-1)
    if factor_end == 9:
        return locs[factor_start - 1 :]
    return locs[factor_start - 1 : factor_end - 9]


def integer_log_space125(exp_start: int, exp_end: int) -> np.typing.NDArray:
    magnitudes = np.logspace(exp_start, exp_end, (exp_end - exp_start) + 1)
    factors = [1, 2, 5]
    return (np.tile(factors, len(magnitudes)) * np.repeat(magnitudes, len(factors)))[:-2]


ROOT_NAMESPACE_UUID = UUID("d0996f2c-be1f-4d09-9b1f-0ec5ff7ab6ca")
