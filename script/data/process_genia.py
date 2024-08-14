"""
Convert GENIA to document-level for prompt input.
"""

import re
from sciriff.lib import util
from pathlib import Path


def create_org_text(sentences):
    text = " ".join([" ".join(sentence) for sentence in sentences])
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+(-)\s+", r"\1", text)
    return text


def create_coref_answer(sentences, clusters):
    flat_sentences = [token for sentence in sentences for token in sentence]
    coref_answer = []
    for cluster in clusters:
        coref_cluster = tuple(
            " ".join(flat_sentences[start : end + 1]) for start, end in cluster
        )
        coref_answer.append(coref_cluster)
    return coref_answer


def create_ner_answer(sentences, ner_annotations, clusters):
    ner_answer = set()
    flat_sentences = [token for sentence in sentences for token in sentence]
    for sentence_ner_annotations in ner_annotations:
        for start, end, entity_type in sentence_ner_annotations:
            entity_name = " ".join(flat_sentences[start : end + 1])
            ner_answer.add(f"{entity_name}: {entity_type}")
    # assign first mention type to all mentions when annotation is missing
    for cluster in clusters:
        first_mention_type = None
        first_mention_name = " ".join(flat_sentences[cluster[0][0] : cluster[0][1] + 1])
        for mention in ner_answer:
            if first_mention_name in mention:
                first_mention_type = mention.split(": ")[1]
                break
        if first_mention_type:
            for start, end in cluster[1:]:
                entity_name = " ".join(flat_sentences[start : end + 1])
                ner_answer.add(f"{entity_name}: {first_mention_type}")

    return list(ner_answer)


def transform_data(data_sample):
    org_text = create_org_text(data_sample["sentences"])
    ner_answer = create_ner_answer(
        data_sample["sentences"], data_sample["ner"], data_sample["clusters"]
    )
    coref = create_coref_answer(data_sample["sentences"], data_sample["clusters"])

    # post-processing
    org_text = util.fix_parentheses(util.fix_quotes(util.fix_percent_symbol(org_text)))
    ner_answer = util.post_process_in_list(ner_answer)
    coref = util.post_process_in_list(coref)

    new_data_sample = {
        "org_text": org_text,
        "ner_answer": ner_answer,
        "coref": coref,
        "re_answer": [],
    }
    return new_data_sample


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads")
    out_dir = Path("data/preprocessing/processed")

    for fold in ["train", "validation", "test"]:
        data = util.load_jsonl(in_dir / f"genia/{fold}.json")
        transformed_data = [transform_data(sample) for sample in data]
        util.write_jsonl(transformed_data, out_dir / f"genia/{fold}.jsonl")
