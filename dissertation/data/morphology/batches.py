from pathlib import Path

BATCHES_DIR = Path(__file__).parent / "batches"

BATCHES = {d.name: list(d.glob("*.data")) for d in BATCHES_DIR.glob("*") if d.is_dir()}

BATCHES["all"] = [f for d in BATCHES.values() for f in d]
