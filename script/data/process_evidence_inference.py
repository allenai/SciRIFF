"""
Convert to .jsonl so that it plays nicely with Huggingface loader.
"""

import pandas as pd
import ast
from pathlib import Path
import re

from sciriff.lib import util


def extract_label_field(s):
    # Define the pattern for the LABEL field
    label_pattern = r"\[LABEL\] (.*?) \[OUT\]"

    # Extract the LABEL value using the regular expression
    match = re.search(label_pattern, s)
    return match.group(1) if match else None


def format_output_entry(output_entry):
    label = extract_label_field(output_entry[-1])
    if label not in [
        "significantly increased",
        "significantly decreased",
        "no significant difference",
    ]:
        raise ValueError("Invalid outcome")

    # Format by ordering as [intervention, comparator, outcome, effect, evidence].
    formatted = [
        output_entry[0],
        output_entry[2],
        output_entry[1],
        label,
        output_entry[3],
    ]

    return formatted


def format_output(raw_output):
    res = []
    output_list = ast.literal_eval(raw_output)
    for entry in output_list:
        to_append = format_output_entry(entry)
        res.append(to_append)

    return res


def format_fold(df):
    instances = []
    for _, row in df.iterrows():
        the_input = row["X"]
        the_output = format_output(row["Y"])
        instance = {"document": the_input, "evidence": the_output}
        instances.append(instance)

    return instances


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads/evidence_inference")
    out_dir = Path("data/preprocessing/processed/evidence_inference")
    out_dir.mkdir(exist_ok=True)

    fold_map = {"train": "train", "dev": "validation", "test": "test"}

    for fold in ["train", "dev", "test"]:
        df = pd.read_csv(in_dir / f"{fold}_wlabels.csv")
        res = format_fold(df)
        fold_name = fold_map[fold]
        out_name = out_dir / f"{fold_name}.jsonl"
        util.write_jsonl(res, out_name)
