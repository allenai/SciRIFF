"""
Utility functions
"""

import json
import re
import os
import stat

# jsonl file handling.


def load_jsonl(fname):
    """
    Load a jsonl file.
    """
    with open(fname) as f:
        return [json.loads(line) for line in f]


def write_jsonl(xs, fname):
    """
    Write a list of dicts to a jsonl file.
    """
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)


def write_protect(fname):
    """
    Write-protect a file.
    """
    os.chmod(fname, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# Data structures


def list_to_dict(xs, key):
    """
    Convert a list of dicts into a dict, keyed by `key`.
    """
    return {x[key]: x for x in xs}


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


# Simple math functions.


def mean(xs):
    return sum(xs) / len(xs)


# Argument parsing


def comma_separated_list_arg(string):
    return [int(x) for x in string.split(",")]


# Dataset processing


def fix_parentheses(text):
    # replace -LRB- and -RRB- with parentheses and properly handle spaces
    text = re.sub(r"\s?-LRB-\s?", "(", text)
    text = re.sub(r"\s?-RRB-\s?", ")", text)

    # add a space after every ) unless it is followed by a punctuation
    text = re.sub(r"\)([^\s.,;:])", r") \1", text)

    # add a space before every ( unless it is preceded by a space
    text = re.sub(r"([^\s])\(", r"\1 (", text)

    # remove extra spaces before "(" and after ")"
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def fix_quotes(text):
    # fix some edge cases of quotes
    text = re.sub(r"\`\`\s?(.*?)\s?''", r"'\1'", text)
    text = re.sub(r"''\s?(.*?)\s?''", r"'\1'", text)

    # remove any space between the single quotes and the text inside them
    text = re.sub(r"' (.*?) '", r"'\1'", text)
    return text


def fix_percent_symbol(text):
    # remove space between number and %
    text = re.sub(r"(\d+)\s+%", r"\1%", text)
    return text


def post_process_in_list(data_list):
    def apply_fixes(item):
        if isinstance(item, str):
            return fix_parentheses(fix_quotes(fix_percent_symbol(item)))
        elif isinstance(item, list) or isinstance(item, tuple):
            return [apply_fixes(subitem) for subitem in item]
        else:
            return item

    return [apply_fixes(item) for item in data_list]
