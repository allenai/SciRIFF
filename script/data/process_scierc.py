"""
Convert SciERC to document-level for prompt input.
"""

import re
from sciriff.lib import util
from pathlib import Path


def create_org_text(sentences):
    text = " ".join([" ".join(sentence) for sentence in sentences])
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+(-)\s+", r"\1", text)
    return text


def create_ner_answer(sentences, ner_annotations, clusters):
    ner_answer = set()
    flat_sentences = [token for sentence in sentences for token in sentence]
    flat_ner_annotations = [
        annotation for sublist in ner_annotations for annotation in sublist
    ]
    span_to_type = {
        tuple(ner_annotation[:2]): ner_annotation[2]
        for ner_annotation in flat_ner_annotations
    }
    mention_to_first_mention = {}
    for cluster in clusters:
        if cluster:
            first_mention_span = tuple(cluster[0])
            first_mention_text = " ".join(
                flat_sentences[first_mention_span[0] : first_mention_span[1] + 1]
            )
            first_mention_type = span_to_type.get(first_mention_span, "Unknown")
            for start, end in cluster:
                mention_text = " ".join(flat_sentences[start : end + 1])
                mention_to_first_mention[mention_text] = (
                    first_mention_text,
                    first_mention_type,
                )
    for cluster in clusters:
        if cluster:
            start, end = cluster[0]
            entity_type = span_to_type.get((start, end), "Unknown")
            entity_name = " ".join(flat_sentences[start : end + 1])
            ner_answer.add((entity_name, entity_type))
    for ner_annotation in flat_ner_annotations:
        start, end, entity_type = ner_annotation
        entity_name = " ".join(flat_sentences[start : end + 1])
        if entity_name not in mention_to_first_mention:
            ner_answer.add((entity_name, entity_type))
    return list(ner_answer)


def create_re_answer(sentences, ner_annotations, relations, clusters):
    re_answer = set()
    flat_sentences = [token for sentence in sentences for token in sentence]
    flat_ner_annotations = [
        annotation for sublist in ner_annotations for annotation in sublist
    ]
    span_to_type = {
        tuple(ner_annotation[:2]): ner_annotation[2]
        for ner_annotation in flat_ner_annotations
    }
    mention_to_first_mention = {}
    for cluster in clusters:
        if cluster:
            first_mention_span = tuple(cluster[0])
            first_mention_text = " ".join(
                flat_sentences[first_mention_span[0] : first_mention_span[1] + 1]
            )
            first_mention_type = span_to_type.get(first_mention_span, "Unknown")
            for start, end in cluster:
                mention_text = " ".join(flat_sentences[start : end + 1])
                mention_to_first_mention[mention_text] = (
                    first_mention_text,
                    first_mention_type,
                )
    for relation in relations:
        for start1, end1, start2, end2, rel_type in relation:
            entity1_text = " ".join(flat_sentences[start1 : end1 + 1])
            entity2_text = " ".join(flat_sentences[start2 : end2 + 1])
            entity1_type = span_to_type.get((start1, end1), "Unknown")
            entity2_type = span_to_type.get((start2, end2), "Unknown")
            entity1_text, entity1_type = mention_to_first_mention.get(
                entity1_text, (entity1_text, entity1_type)
            )
            entity2_text, entity2_type = mention_to_first_mention.get(
                entity2_text, (entity2_text, entity2_type)
            )
            re_answer.add((entity1_text, rel_type, entity2_text))
    return list(re_answer)


def create_coref(sentences, clusters):
    coref = []
    flat_sentences = [token for sentence in sentences for token in sentence]
    for cluster in clusters:
        coref_cluster = []
        for start, end in cluster:
            mention = " ".join(flat_sentences[start : end + 1])
            coref_cluster.append(mention)
        if len(coref_cluster) > 1:
            coref.append(tuple(coref_cluster))
    return coref


def transform_data(data_sample):
    org_text = create_org_text(data_sample["sentences"])
    ner_answer = create_ner_answer(
        data_sample["sentences"], data_sample["ner"], data_sample["clusters"]
    )
    re_answer = create_re_answer(
        data_sample["sentences"],
        data_sample["ner"],
        data_sample["relations"],
        data_sample["clusters"],
    )
    coref = create_coref(data_sample["sentences"], data_sample["clusters"])

    # post-processing

    org_text = util.fix_parentheses(util.fix_quotes(util.fix_percent_symbol(org_text)))
    ner_answer = util.post_process_in_list(ner_answer)
    re_answer = util.post_process_in_list(re_answer)
    coref = util.post_process_in_list(coref)

    new_data_sample = {
        "org_text": org_text,
        "ner_answer": ner_answer,
        "re_answer": re_answer,
        "coref": coref,
    }
    return new_data_sample


if __name__ == "__main__":
    in_dir = Path("data/preprocessing/downloads")
    out_dir = Path("data/preprocessing/processed")

    fold_map = {"train": "train", "dev": "validation", "test": "test"}

    for fold in ["train", "dev", "test"]:
        data = util.load_jsonl(in_dir / f"scierc/{fold}.json")
        transformed_data = [transform_data(sample) for sample in data]
        util.write_jsonl(transformed_data, out_dir / f"scierc/{fold_map[fold]}.jsonl")
