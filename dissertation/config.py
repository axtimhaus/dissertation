from pathlib import Path

ROOT_DIR = Path(__file__).parent

import jinja2

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))
