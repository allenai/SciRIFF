"""
Convert QASA to required format.
"""
import json
from sciriff.lib import util
from pathlib import Path


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads")
    out_dir = Path("data/preprocessing/processed")

    with open(in_dir / "qasa/qasa_test_v1.1.json", "r") as fin:
        qasa_dataset = json.load(fin)
    util.write_jsonl(
        [qa for ix, qa in qasa_dataset.items() if not qa.update({"qix": ix})],
        out_dir / "qasa/test.jsonl",
    )
