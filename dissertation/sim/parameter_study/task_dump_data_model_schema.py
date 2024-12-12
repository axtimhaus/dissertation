import json
from pathlib import Path

THIS_DIR = Path(__file__).parent

def task_dump_data_model_schema(src_file = THIS_DIR / "data_model.py", produces = THIS_DIR / "ParameterStudy.schema.json" ):
    from dissertation.sim.parameter_study.data_model import ParameterStudy

    schema = ParameterStudy.model_json_schema()
    produces.write_text(json.dumps(schema, indent=4))
