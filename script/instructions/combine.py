"""
Combine all tasks into a single dataset. Remove unnecessary metadata.
"""

import argparse
from pathlib import Path
from sciriff.lib import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=Path,
        help="Directory with one subdir per task, created by `instantiate.py`.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Output directory with all tasks combined. One file per split.",
    )

    return parser.parse_args()


def format_inst(inst):
    metadata_fields = [
        "domains",
        "input_context",
        "output_context",
        "source_type",
    ]
    metadata = {k: inst["metadata"][k] for k in metadata_fields}
    metadata["task_family"] = inst["metadata"]["task"]
    res = {
        "input": inst["input"],
        "output": inst["output"],
        "metadata": metadata,
        "_instance_id": inst["_instance_id"],
    }
    return res


def main():
    res = {"train": [], "validation": [], "test": []}

    args = get_args()
    args.out_dir.mkdir(exist_ok=True)

    for task_dir in args.in_dir.iterdir():
        for split in ["train", "validation", "test"]:
            insts = util.load_jsonl(task_dir / f"{split}.jsonl")
            formatted = [format_inst(inst) for inst in insts]
            res[split].extend(formatted)

    res = {
        split: sorted(v, key=lambda x: x["_instance_id"]) for split, v in res.items()
    }
    for split, v in res.items():
        util.write_jsonl(v, args.out_dir / f"{split}.jsonl")


if __name__ == "__main__":
    main()
