"""
Do both extractive and abstractive questions. Truncate the input document at a fixed
context window.
"""

from transformers import LlamaTokenizerFast
import random


def transform_dict(data, keys=None):
    # Go from a dict of lists to a list of dicts.
    if keys is not None:
        data = {key: data[key] for key in keys}
    else:
        keys = list(data.keys())

    # All entries should have the same length.
    if len(set([len(data[key]) for key in keys])) > 1:
        raise ValueError("All keys should have same length.")

    return [dict(zip(keys, values)) for values in zip(*data.values())]


def sum_field(data, field):
    """
    Sum a field across all entries in a list of dicts.
    """
    return sum([entry[field] for entry in data])


class Formatter:
    def __init__(
        self,
        question_type,
        tokenizer_path="meta-llama/Llama-2-7b-hf",
        context_window=4096,
        instruction_tokens=300,  # Number of tokens to allocate for task instructions.
    ):
        if question_type not in ["extractive", "abstractive"]:
            raise ValueError("Invalid question type.")

        self.question_type = question_type
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
        self.context_window = context_window
        self.rng = random.Random(76)
        self.instruction_tokens = instruction_tokens

    def count_toks(self, text):
        """
        Count the number of tokens in text.
        """
        return len(self.tokenizer.tokenize(text))

    def format_doc(self, entry):
        """
        Format a document and count tokens.
        """
        transformed_text = transform_dict(entry["full_text"])

        # Format title and abstract
        title = {"content": entry["title"], "n_toks": self.count_toks(entry["title"])}
        abstract = {
            "content": entry["abstract"],
            "n_toks": self.count_toks(entry["abstract"]),
        }

        # Format full text.
        full_text = []
        block_ix = 0
        for block in transformed_text:
            # Skip block if empty.
            if block["paragraphs"] == [""]:
                continue
            section_name = (
                "Unnamed section"
                if block["section_name"] is None
                else block["section_name"]
            )
            paras = [para for para in block["paragraphs"] if para != ""]
            this_segment = section_name + "\n" + "\n".join(paras)
            to_append = {
                "content": this_segment,
                "n_toks": self.count_toks(this_segment),
                "ix": block_ix,
                "type": "full_text",
            }
            full_text.append(to_append)
            block_ix += 1

        # Format captions.
        captions = []
        for cap_ix, cap in enumerate(entry["figures_and_tables"]["caption"]):
            to_append = {
                "content": cap,
                "n_toks": self.count_toks(cap),
                "ix": cap_ix,
                "type": "caption",
            }
            captions.append(to_append)

        res = {
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
            "captions": captions,
        }

        return res

    def get_evidence_section(self, ev, doc):
        """
        Get evidence section for a single piece of evidence.
        """
        for sec in doc["full_text"]:
            if ev in sec["content"]:
                return sec

        # Get rid of the prefix "FLOAT SELECTED: ".
        ev_no_float = ev.replace("FLOAT SELECTED: ", "")
        for cap in doc["captions"]:
            if ev_no_float in cap["content"]:
                return cap

        # If we get here, we didn't find the section.
        return None

    def get_evidence_sections(self, evidence, doc):
        """
        Get the sections of the document containing the evidence.
        """
        res = []

        for ev in evidence:
            ev_sec = self.get_evidence_section(ev, doc)

            # If we missed evidence, return None.
            if ev_sec is None:
                return None

            # There may be times when the same section is referenced multiple times.
            if ev_sec not in res:
                res.append(ev_sec)

        return res

    def format_input_doc(self, doc, evidence_secs, distractors):
        """
        Format input document given list of evidence sections and distractors.
        """
        # Need to put the full text before the captions.
        type_order = {"full_text": 0, "caption": 1}
        body_content = sorted(
            evidence_secs + distractors, key=lambda x: (type_order[x["type"]], x["ix"])
        )
        all_content = [doc["title"], doc["abstract"]] + body_content
        content_text = [x["content"] for x in all_content]

        input_doc = "\n\n".join(content_text)
        return input_doc

    def make_input_doc(self, evidence_secs, doc):
        """
        Create input document for this answer. Do this by padding the evidence sections
        with distractors until we hit the token limit.
        """
        candidate_blocks = [
            block for block in doc["full_text"] if block not in evidence_secs
        ]
        candidate_captions = [
            cap for cap in doc["captions"] if cap not in evidence_secs
        ]
        candidates = candidate_captions + candidate_blocks

        # Make sure we have the right number of candidates.
        expected_len = len(doc["full_text"]) + len(doc["captions"]) - len(evidence_secs)
        if len(candidates) != expected_len:
            raise ValueError("Something went wrong.")

        self.rng.shuffle(candidates)

        # Total number of tokens so far.
        token_total = (
            self.instruction_tokens
            + doc["title"]["n_toks"]
            + doc["abstract"]["n_toks"]
            + sum_field(evidence_secs, "n_toks")
        )

        # Add distractors till we hit the token limit.
        distractors = []
        if token_total < self.context_window:
            while True and len(candidates):
                distractor = candidates.pop()
                token_total = token_total + distractor["n_toks"]
                if token_total > self.context_window:
                    # If this distractor would put us over the limit, stop.
                    break
                else:
                    # Otherwise, keep adding more distractors.
                    distractors.append(distractor)

        input_doc = self.format_input_doc(doc, evidence_secs, distractors)

        return input_doc

    def format_answer(self, question, answer, doc):
        """
        Format a single answer. This requires grabbing all the relevant evidence and
        then padding with distractors.
        """
        # If the highlighted evidence contains an INLINEFORM, then the question might
        # not be answerable. Skip these.
        for highlight in answer["highlighted_evidence"]:
            if "inlineform" in highlight.lower():
                return None

        # If it uses figures or tables, skip; can't answer these from paper body.
        evidence = [
            ev for ev in answer["evidence"] if "FLOAT SELECTED" not in ev
        ]
        # If all evidence was in figures or tables, don't use this one.
        if not evidence:
            return None

        evidence_secs = self.get_evidence_sections(evidence, doc)
        # If we couldn't find evidence, return None.
        if evidence_secs is None:
            return None

        paper = self.make_input_doc(evidence_secs, doc)

        res = {"paper": paper, "question": question}

        # Skip yes / no questions.
        if answer["yes_no"]:
            return None
        if answer["unanswerable"]:
            res["unanswerable"] = True
            res["answer"] = None
            res["evidence"] = None
            return res

        if answer["free_form_answer"] != "":
            # If we're doing extractive QA, skip.
            if self.question_type == "extractive":
                return None
            res["unanswerable"] = False
            res["answer"] = answer["free_form_answer"]
            res["evidence"] = evidence
            return res
        if answer["extractive_spans"] != []:
            # If we're doing abstractive QA, skip.
            if self.question_type == "abstractive":
                return None
            res["unanswerable"] = False
            res["answer"] = answer["extractive_spans"]
            res["evidence"] = evidence
            return res

        # If there's no match, return None.
        return None

    @staticmethod
    def get_answers(qa):
        # In some cases, there are multiple answers of a given type (abstractive /
        # extractive) for a single question. Just keep the first one.
        answers = []
        seen_extractive = False
        seen_abstractive = False

        for answer in qa["answers"]["answer"]:
            if answer["extractive_spans"] != [] and not seen_extractive:
                answers.append(answer)
                seen_extractive = True
            if answer["free_form_answer"] != "" and not seen_abstractive:
                answers.append(answer)
                seen_abstractive = True

        return answers

    def format_instance(self, entry):
        """
        Format all QA pairs for a single document.
        """
        doc = self.format_doc(entry)
        qas = transform_dict(entry["qas"], keys=["question", "answers"])

        instances = []
        id_counter = 0
        for qa in qas:
            question = qa["question"]
            answers = self.get_answers(qa)
            for answer in answers:
                formatted_instance = self.format_answer(question, answer, doc)
                # Check case where the answer doesn't match any categories.
                if formatted_instance is not None:
                    # For qasper the instance and the subquestion in the ID.
                    formatted_instance["_id"] = f"{entry['_id']}.{id_counter}"
                    id_counter += 1
                    instances.append(formatted_instance)

        return instances

    def format_instances(self, dataset):
        instances = []
        for instance in dataset:
            instances.extend(self.format_instance(instance))

        return instances
