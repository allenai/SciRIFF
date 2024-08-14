"""
Make sure the inputs and outputs look well-behaved. Throw an error if problems.

- Make sure there's no empty inputs or outputs.
- Make sure all outputs that are listed as `json` are actually json-parseable.
- Similarly, for `jsonlines` every line should be json-parseable.

Usage: pytest check_instances.py
"""

from sciriff.instructions import task
import json
from sciriff.lib import paths, util
from tqdm import tqdm
import pytest

template_collection = task.TaskCollection()


def check_instance(instance, template):
    assert len(instance["input"]) >= 5
    assert instance["input"] != ""
    assert instance["output"] != ""

    if template.metadata["output_context"] == "json":
        # If this throws an error, there's a problem.
        try:
            _ = json.loads(instance["output"])
        except Exception:
            assert False

    if template.metadata["output_context"] == "jsonlines":
        lines = instance["output"].splitlines()
        for line in lines:
            try:
                _ = json.loads(line)
            except Exception:
                assert False


def test_instances():
    for subdir in paths.INSTANCE_DIR.iterdir():
        print(f"Checking {subdir.name}.")
        n_dirs = len(list(subdir.iterdir()))
        for task_path in tqdm(subdir.iterdir(), total=n_dirs):
            task_name = task_path.name
            template = template_collection.tasks[task_name]

            # Check make sure the right number of folds are there.
            assert len(list(task_path.iterdir())) == 3

            for fold_file in task_path.iterdir():
                data = util.load_jsonl(fold_file)
                for instance in data:
                    check_instance(instance, template)
