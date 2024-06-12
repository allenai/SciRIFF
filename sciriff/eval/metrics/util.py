"""
Utils shared by different metrics.
"""

import re
import string
from collections import Counter


def safe_div(x, y):
    return x / y if y != 0 else 0


def sum_dict(d):
    "Sum all values in a dict."
    return sum(d.values())


def count_dict(keys):
    "Initialize a count dict with a fixed set of keys."
    # This is different from `Counter` because it breaks if unknown keys are given.
    return {k: 0 for k in keys}


def normalize_list_entry(entry):
    "Convert to lowercase and convert numbers to strings."
    if entry is None:
        return ""
    elif isinstance(entry, str):
        return entry.strip().lower()
    elif isinstance(entry, int):
        return str(entry)
    elif isinstance(entry, float):
        return str(round(entry, 2))
    elif isinstance(entry, list):
        # If the model wrote a nested list, flatten and join on whitespace.
        return " ".join(flatten(entry))
    else:
        raise TypeError("Unexpected type for list entry.")


def compute_f1(correct, preds, refs):
    precision = correct / preds if preds > 0 else 0
    recall = correct / refs if refs > 0 else 0
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {"p": precision, "r": recall, "f1": f1}


def flatten(nested_list):
    """
    Flatten a (possibly nested) list.
    """
    flat_list = []

    # Function to flatten each element
    def flatten_element(element):
        if isinstance(element, list):
            for item in element:
                flatten_element(item)
        else:
            flat_list.append(element)

    # Flatten each element in the nested list
    for element in nested_list:
        flatten_element(element)

    return flat_list


# Token-level F1 like in Squad

# I copied these from https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py

def token_f1_normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """
    # Called `normalize_answer` in Qasper source file.

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(pred, ref):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    # Called `token_f1_score` in Qasper source file.
    prediction_tokens = token_f1_normalize_answer(pred).split()
    reference_tokens = token_f1_normalize_answer(ref).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
