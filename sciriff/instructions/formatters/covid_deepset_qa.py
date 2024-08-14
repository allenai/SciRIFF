import nltk
import random
from transformers import LlamaTokenizerFast


random.seed(42)


class Formatter:
    """
    Format covid_qa_deepset from HF. This is a simple formatter to construct output data structures.
    """

    def __init__(
        self,
        tokenizer_path="meta-llama/Llama-2-7b-hf",
        context_window=4096,
        instruction_tokens=200,
    ):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
        self.context_window = context_window
        self.instruction_tokens = instruction_tokens

    def count_toks(self, text):
        """
        Count the number of tokens in text.
        """
        return len(self.tokenizer.tokenize(text))

    def format_instance(self, instance):
        length_formatted_context = []

        context = instance["context"]
        answer_text = " ".join(instance["answers"]["text"])
        if context.find(answer_text) > -1:
            answer_context = context.split(answer_text)
            if context.find(answer_text) == 0:
                before_answer_context = {}
                answer_ix = 0
                after_answer_context = {
                    ix: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(nltk.sent_tokenize(answer_context[0]))
                }
            elif len(answer_context) == 2:
                before_answer_context = {
                    ix: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(nltk.sent_tokenize(answer_context[0]))
                }
                answer_ix = len(before_answer_context)
                after_answer_context = {
                    ix
                    + answer_ix
                    + 1: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(nltk.sent_tokenize(answer_context[1]))
                }
            elif len(answer_context) < 2:
                before_answer_context = {
                    ix: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(nltk.sent_tokenize(answer_context[0]))
                }
                answer_ix = len(before_answer_context)
                after_answer_context = {}
            else:
                before_answer_context = {
                    ix: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(nltk.sent_tokenize(answer_context[0]))
                }
                answer_ix = len(before_answer_context)
                after_answer_context = {
                    ix
                    + answer_ix
                    + 1: {"sent": sent, "sent_size": self.count_toks(sent)}
                    for ix, sent in enumerate(
                        nltk.sent_tokenize(" ".join(answer_context[1:]))
                    )
                }
        else:
            raise ValueError("Answer not found in the context.")

        length_formatted_context = [answer_text]
        context_length = self.count_toks(answer_text) + self.instruction_tokens

        # Pad the answer with atleast one preceeding sentence to add context.
        if len(before_answer_context) > 0:
            length_formatted_context = [
                before_answer_context[answer_ix - 1]["sent"]
            ] + length_formatted_context
            context_length += before_answer_context[answer_ix - 1]["sent_size"]
            before_answer_context.pop(answer_ix - 1)

        candidate_context = [ix for ix in before_answer_context] + [
            ix for ix in after_answer_context
        ]

        stop_increasing_context_size = False
        # We want to include a chunk around the answer, which could be sentences before or after the answer.
        # To ensure that the answer occurs at various positions in the selected context, we randomly sample from the sentences set and start including sentences untill we either reach the context window size or hit the randomly sampled sentence index.
        while not stop_increasing_context_size and len(candidate_context) > 0:
            candidate_ix = random.sample(candidate_context, 1)[0]
            # start from the sentence nearest to the answer and include sentences till we reach context size or the candidate_ix we sampled.
            if candidate_ix > answer_ix:
                for include_sent_ix in range(
                    min(after_answer_context.keys()), candidate_ix + 1
                ):
                    if (
                        after_answer_context[include_sent_ix]["sent_size"]
                        + context_length
                    ) <= self.context_window:
                        length_formatted_context = length_formatted_context + [
                            after_answer_context[include_sent_ix]["sent"]
                        ]
                        context_length += after_answer_context[include_sent_ix][
                            "sent_size"
                        ]
                        after_answer_context.pop(include_sent_ix)
                        candidate_context.remove(include_sent_ix)
                    else:
                        stop_increasing_context_size = True
                        break
            elif candidate_ix < answer_ix:
                for include_sent_ix in range(
                    max(before_answer_context.keys()), (candidate_ix - 1), -1
                ):
                    if (
                        before_answer_context[include_sent_ix]["sent_size"]
                        + context_length
                    ) <= self.context_window:
                        length_formatted_context = [
                            before_answer_context[include_sent_ix]["sent"]
                        ] + length_formatted_context
                        context_length += before_answer_context[include_sent_ix][
                            "sent_size"
                        ]
                        before_answer_context.pop(include_sent_ix)
                        candidate_context.remove(include_sent_ix)
                    else:
                        stop_increasing_context_size = True
                        break
            else:
                raise ValueError(
                    "Answer should have been already included in the context."
                )

        instance["article"] = " ".join(length_formatted_context)
        instance["answer"] = answer_text
        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
