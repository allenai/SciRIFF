import nltk
import random
import string


random.seed(42)


class Formatter:
    """
    Format scientific dataset MCQ (multiple correct answers) task.
    """

    def __init__(
        self,
        num_sents_per_cand=2,  # Number of sentences from abstract of the dataset candidates to be included in the context.
        max_cands=7,
        max_pos_cands=5,
    ):
        self.num_sents_per_cand = num_sents_per_cand
        self.max_cands = max_cands
        self.max_pos_cands = max_pos_cands

    def format_instance(self, instance):
        random.shuffle(instance["positive_cands"])
        random.shuffle(instance["negative_cands"])
        # Sample correct candidates
        positive_options = random.sample(
            instance["positive_cands"],
            random.randint(2, min(len(instance["positive_cands"]), self.max_pos_cands)),
        )
        # Sample negative candidates
        total_options = random.randint(self.max_pos_cands, self.max_cands)
        negative_sample_len = min(
            total_options - len(positive_options), len(instance["negative_cands"])
        )
        if negative_sample_len > 0:
            options = (
                random.sample(instance["negative_cands"], negative_sample_len)
                + positive_options
            )
        else:
            options = (
                positive_options  # The options consists of all positive candidates
            )
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

        # Formulate the correct answer string
        instance["answer"] = []
        for cand in options:
            if cand in positive_options:
                instance["answer"].append(cand)
        instance["answer"] = ", ".join(instance["answer"])

        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
