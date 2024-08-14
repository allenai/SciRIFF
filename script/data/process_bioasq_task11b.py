"""
Format BioASQ_task11B for prompt input.
"""

import json
import random
import numpy as np
from collections import defaultdict
from sciriff.lib import util
from pathlib import Path
from thefuzz import fuzz


random.seed(42)
np.random.seed(42)


def format_binary_qa(qa_dict):
    all_instances = []

    for _, qa in enumerate(qa_dict):
        snippets = "\n".join(
            [snip["text"].replace("\n", " ").strip() for snip in qa["snippets"]]
        )
        question = qa["body"]
        binary_ans = qa["exact_answer"]
        long_ans = " ".join(qa["ideal_answer"])

        data_json = {
            "snippets": snippets,
            "question": question,
            "binary_ans": binary_ans,
            "long_ans": long_ans,
        }
        all_instances.append(data_json)

    return all_instances


def format_list_qa(qa_dict):
    all_instances = []

    for _, qa in enumerate(qa_dict):
        snippets = "\n".join(
            [snip["text"].replace("\n", " ").strip() for snip in qa["snippets"]]
        )
        question = qa["body"]
        correct_answer_list = [x[0] for x in qa["exact_answer"]]
        csv_ans = ", ".join(correct_answer_list)

        student_answer = csv_ans
        student_eval = "Yes, the list is exhaustive and correct."

        data_json = {
            "snippets": snippets,
            "question": question,
            "correct_ans": csv_ans,
        }
        all_instances.append(data_json)

    return all_instances


def format_factoid_qa(qa_dict):
    all_instances = []

    for _, qa in enumerate(qa_dict):
        if type(qa["exact_answer"][0]) == list:
            flattened_list = [
                item for sublist in qa["exact_answer"] for item in sublist
            ]
            qa_dict[_]["exact_answer"] = flattened_list

    for _, qa in enumerate(qa_dict):
        snippets = "\n".join(
            [snip["text"].replace("\n", " ").strip() for snip in qa["snippets"]]
        )
        question = qa["body"]
        correct_answer = max(qa["exact_answer"], key=len)

        data_json = {
            "snippets": snippets,
            "question": question,
            "correct_ans": correct_answer,
        }
        all_instances.append(data_json)

    return all_instances


def format_general_qa(qa_dict):
    all_instances = []

    for _, qa in enumerate(qa_dict):
        snippets = "\n".join(
            [snip["text"].replace("\n", " ").strip() for snip in qa["snippets"]]
        )
        question = qa["body"]
        correct_answer = max(qa["ideal_answer"], key=len)

        if correct_answer.lower() == snippets.lower():
            # we skip instances where the context is exactlty copied as the answer in
            # the dataset. Comprises around 2% of the dataset.
            continue

        if fuzz.ratio(snippets, correct_answer) > 80:
            correct_answer = min(qa["ideal_answer"], key=len)

        data_json = {
            "snippets": snippets,
            "question": question,
            "answer": correct_answer,
        }
        all_instances.append(data_json)

    return all_instances


def format_bioasq_11b_data(in_dir, out_dir, fold):
    qa_type_processor = {
        "yesno": format_binary_qa,
        "list": format_list_qa,
        "factoid": format_factoid_qa,
        "summary": format_general_qa,
    }

    instances = defaultdict(list)
    for fin in list(in_dir.glob(f"{fold}/*.json")):
        qa_dicts = json.load(open(fin, "r"))
        for qa in qa_dicts["questions"]:
            instances[qa["type"]].append(qa)

    for qtype in instances:
        formatted_instances = qa_type_processor[qtype](instances[qtype])
        output_file = out_dir / f"bioasq_task11b_{qtype}_qa/{fold}.jsonl"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        util.write_jsonl(formatted_instances, output_file)
    return


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads")
    out_dir = Path("data/preprocessing/processed")

    for fold in ["train", "test"]:
        formatted_data = format_bioasq_11b_data(
            in_dir / "bioasq_task11b/", out_dir, fold
        )
