"""
Store paths to shared directories.
"""


from pathlib import Path


project_root = Path(__file__).parent.parent.parent

EVAL_DIR = project_root / "sciriff/eval"
RESULT_DIR = project_root / "results"
