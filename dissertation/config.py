from pathlib import Path

ROOT_DIR = Path(__file__).parent
BUILD_DIR = ROOT_DIR / ".build"

import jinja2

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))
