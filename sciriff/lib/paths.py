"""
Store paths to shared directories.
"""


from pathlib import Path


project_root = Path(__file__).parent.parent.parent

EVAL_DIR = project_root / "sciriff/eval"
RESULT_DIR = project_root / "results"
TEMPLATE_DIR = project_root / "sciriff/instructions/templates"
TASK_SCHEMA_FILE = project_root / "sciriff/instructions/task_schema.yaml"
DATA_DIR = project_root / "data"
INSTANCE_DIR = DATA_DIR / "instructions"
