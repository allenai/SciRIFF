import nltk
import random
import string


class Formatter:
    """
    Format scientific dataset MCQ task.
    """

    def __init__(
        self,
        num_sents_per_cand=2,  # Number of sentences from abstract of the dataset candidates to be included in the context.
    ):
        self.num_sents_per_cand = num_sents_per_cand

    def format_instance(self, instance):
        random.shuffle(instance["negative_cands"])
        # Sample three negatives and a positive to create candidates for MCQ
        options = random.sample(instance["negative_cands"], 3) + [instance["answer"]]
        random.shuffle(options)

        options_text = "  ".join(
            [f"{_}: {op}" for _, op in zip(string.ascii_uppercase, options)]
        )
        instance["options"] = options_text

        # Create short context by using the first two lines from abstract of each candidate.
        instance["abstracts"] = {
            entry["dkey"]: entry["dval"].replace("\n", " ")
            for entry in instance["abstracts"]
        }
        context = ""
        for candidate in options:
            context += (
                candidate
                + ": "
                + " ".join(
                    nltk.sent_tokenize(instance["abstracts"][candidate])[
                        0 : self.num_sents_per_cand
                    ]
                )
                + "\n"
            )
        instance["context"] = context.strip()
        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
