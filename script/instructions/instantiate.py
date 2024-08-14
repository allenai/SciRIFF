"""
Instantiate datasets from templates.
"""

import argparse
import json
from datasets.utils.logging import disable_progress_bar

from sciriff.lib import util
from sciriff.instructions.instantiator import Instantiator


def write_jsonl(xs, fname):
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)


def get_args():
    parser = argparse.ArgumentParser(description="Instantiate datasets from templates.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="A comma-separated list of tasks. If not given, do all available.",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=None,
        help=(
            "Randomly choose `n_instances` per task split. If not given, do all. "
            "For tasks with less than `n_instances`, use all."
        ),
    )
    parser.add_argument(
        "--templates",
        type=util.comma_separated_list_arg,
        default=None,
        help="A comma-separated list of template ID's. If not given, do all.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. By default, tasks/instances.",
    )
    parser.add_argument(
        "--random_template",
        action="store_true",
        default=False,
        help=(
            "If passed, randomly select a template for each instance. If `templates` "
            "is specified, select from the given templates."
        ),
    )
    parser.add_argument(
        "--no_clobber",
        action="store_true",
        help="If passed, do not overwrite files in existing instance directories.",
    )
    parser.add_argument(
        "--catch_errors",
        action="store_true",
        help="If given, catch errors and continue rather than raising immediately.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "If given, parallelize over this many worker processes. "
            "This will always catch errors rather than raising immediately."
        ),
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=4096,
        help="Maximum number of tokens per instance, including both input and output.",
    )

    return parser.parse_args()


def main():
    disable_progress_bar()  # Makes huggingface datasets less chatty.
    args = get_args()
    instantiator = Instantiator(args)
    instantiator.instantiate()


if __name__ == "__main__":
    main()
