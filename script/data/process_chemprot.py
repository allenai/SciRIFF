"""
Convert Chemprot to document-level for prompt input.
"""

import re
from sciriff.lib import util
from pathlib import Path


def create_org_text(sentences):
    """Create 'org_text' field by concatenating all sentences."""
    text = " ".join([" ".join(sentence) for sentence in sentences])
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+(-)\s+", r"\1", text)
    return text


def create_ner_answer(sentences, ner_annotations):
    """Create 'ner_answer' field by extracting all unique NER annotations."""
    ner_answer = set()
    flat_sentences = [token for sentence in sentences for token in sentence]
    for sentence_ner_annotations in ner_annotations:
        for start, end, entity_type in sentence_ner_annotations:
            entity_name = " ".join(flat_sentences[start : end + 1])
            ner_answer.add(f"{entity_name}: {entity_type}")
    return list(ner_answer)


def create_re_answer(sentences, relations):
    """Create 're_answer' field by extracting all unique relation annotations."""
    re_answer = set()
    flat_sentences = [token for sentence in sentences for token in sentence]
    for sentence_relations in relations:
        for start1, end1, start2, end2, relation_type in sentence_relations:
            entity1 = " ".join(flat_sentences[start1 : end1 + 1])
            entity2 = " ".join(flat_sentences[start2 : end2 + 1])
            re_answer.add(f"({entity1}, {relation_type}, {entity2})")
    return list(re_answer)


def transform_chemprot_data(data_sample):
    org_text = create_org_text(data_sample["sentences"])
    ner_answer = create_ner_answer(data_sample["sentences"], data_sample["ner"])
    re_answer = create_re_answer(data_sample["sentences"], data_sample["relations"])

    # post-processing
    org_text = util.fix_parentheses(util.fix_quotes(util.fix_percent_symbol(org_text)))
    ner_answer = util.post_process_in_list(ner_answer)
    re_answer = util.post_process_in_list(re_answer)

    new_data_sample = {
        "org_text": org_text,
        "ner_answer": ner_answer,
        "re_answer": re_answer,
        "coref": [],
    }
    return new_data_sample


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads")
    out_dir = Path("data/preprocessing/processed")

    for fold in ["train", "validation", "test"]:
        data = util.load_jsonl(in_dir / f"chemprot/{fold}.json")
        transformed_data = [transform_chemprot_data(sample) for sample in data]
        util.write_jsonl(transformed_data, out_dir / f"chemprot/{fold}.jsonl")
