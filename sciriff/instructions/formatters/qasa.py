import nltk
import string
from transformers import LlamaTokenizerFast


class Formatter:
    """
    Format QASA dataset. The context consists of evidential paragraphs, we need to trim them to specific context_window size.
    We select sentences with no overlap (or till the specified threshold percentage) with the answer and trim the non-relevant sentences if context size exceeds context_window.
    """

    def __init__(
        self,
        tokenizer_path="meta-llama/Llama-2-7b-hf",
        context_window=4096,
        instruction_tokens=200,
        removal_thresh_percentage=0.2,  # The overlap threshold of answer with context_sentences to qualify for removal from context to trim the context.
    ):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
        self.context_window = context_window
        self.instruction_tokens = instruction_tokens
        self.removal_thresh_percentage = removal_thresh_percentage

    def count_toks(self, text):
        """
        Count the number of tokens in text.
        """
        return len(self.tokenizer.tokenize(text))

    def format_instance(self, instance):
        length_formatted_context = []

        all_context = []
        for evidence_para in instance["evidential_info"]:
            all_context.append(evidence_para["context"])
        all_context = " ".join(all_context)
        if self.count_toks(all_context) <= self.context_window:
            instance["context"] = all_context
            return instance

        # Remove sentences with no overlapping content words with the answer to trim the evidential paragraphs to fit context_window size.
        ans_context_overlap = []
        nltk_stop_words = list(nltk.corpus.stopwords.words("english"))
        ans_tokens = nltk.word_tokenize(instance["composition"])
        ans_tokens = list(
            filter(
                lambda token: token not in string.punctuation
                and token.lower() not in nltk_stop_words,
                ans_tokens,
            )
        )
        ans_len = len(set(ans_tokens))

        all_context_sents = nltk.sent_tokenize(all_context)
        for ix, sent in enumerate(all_context_sents):
            sent_tokens = nltk.word_tokenize(sent)
            ans_context_overlap.append(
                {
                    "ix": ix,
                    "sent": sent,
                    "sent_token_len": self.count_toks(sent),
                    "overlap": len(set(ans_tokens).intersection(set(sent_tokens)))
                    / ans_len,
                }
            )

        total_context_length = sum(
            [para["sent_token_len"] for para in ans_context_overlap]
        )

        # Sort to remove the sentences with least overlap with answer and maximum length sentences
        to_remove_sents = []
        ans_context_overlap = sorted(
            ans_context_overlap, key=lambda k: (k["overlap"], -k["sent_token_len"])
        )
        for sent_info in ans_context_overlap:
            if total_context_length < self.context_window:
                break

            if sent_info["overlap"] < self.removal_thresh_percentage:
                to_remove_sents.append(sent_info["ix"])
                total_context_length -= sent_info["sent_token_len"]
            else:
                # Cannot fit the context into context_window without removing substantial answer chunk from the context. Will exclude this instance. To include, either increase the removal threshold or increase the context_window.
                print(
                    f"Excluding instance from {instance['paper_id']}, qix: {instance['qix']} as cannot fit the answer context in appropriate context_window size without removing relevant text."
                )
                return None

        trimmed_context = []
        for sent_info in sorted(ans_context_overlap, key=lambda k: k["ix"]):
            if sent_info["ix"] in to_remove_sents:
                continue
            trimmed_context.append(sent_info["sent"])

        instance["context"] = " ".join(trimmed_context)
        if instance["composition"].startswith("[") and instance["composition"].endswith(
            "]"
        ):
            instance["composition"] = instance["composition"][1:-1]
        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset if instance]
