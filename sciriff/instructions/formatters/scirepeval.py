from transformers import LlamaTokenizerFast
import random
import json
from collections import OrderedDict


class Formatter:
    def __init__(
        self,
        tokenizer_path="meta-llama/Llama-2-7b-hf",
        fields_to_keep=["title", "abstract"],
        context_window=4096,
        instruction_tokens=500,  # Number of tokens to allocate for task instructions.
        abstract_character_limit=2048,  # Number of characters to truncate abstracts to.
    ):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
        self.fields_to_keep = fields_to_keep
        if "id" not in self.fields_to_keep:
            self.fields_to_keep = ["id"] + self.fields_to_keep
        self.context_window = context_window
        self.rng = random.Random(76)
        self.instruction_tokens = instruction_tokens
        self.abstract_character_limit = abstract_character_limit

    def count_toks(self, text):
        """
        Count the number of tokens in text.
        """
        return len(self.tokenizer.tokenize(text))

    def format_instance(self, instance):
        """
        Format a single instance of search and results.
        """
        # truncate the abstracts to string limits
        for candidate in instance["candidates"]:
            if "abstract" in candidate:
                candidate["abstract"] = candidate["abstract"][
                    : self.abstract_character_limit
                ]

        # interleave the candidates from all scores. e.g. if the scores are {2, 1, 0}
        # then we want to interleave the candidates in the order [2, 1, 0, 2, 1, 0, 2, 1, 0]
        candidates = instance["candidates"]
        candidates_by_score = {}
        for candidate in candidates:
            score = candidate["score"]
            if score not in candidates_by_score:
                candidates_by_score[score] = []
            candidates_by_score[score].append(candidate)

        # do the interleaving
        candidates = []
        while len(candidates_by_score) > 0:
            for score in sorted(candidates_by_score.keys(), reverse=True):
                candidates.append(candidates_by_score[score].pop())
                if len(candidates_by_score[score]) == 0:
                    del candidates_by_score[score]

        # then add the "id" field to each candidate for the purpose of computing the number of tokens
        for i, candidate in enumerate(candidates):
            candidate["id"] = f"[{i + 1}]"

        # keep candidates until we get to too many
        running_length = 0
        num_to_keep = 0
        for candidate in candidates:
            # now format the candidates into a string and see if it's too long
            # keeps order and important to only keep the "id", "title", and "abstract" fields
            candidate_dict = OrderedDict((k, candidate[k]) for k in self.fields_to_keep)
            candidate_string = json.dumps(candidate_dict, indent=2)
            length = self.count_toks(candidate_string)
            running_length += length + 1  # +1 for newline
            if running_length > self.context_window - self.instruction_tokens:
                break
            num_to_keep += 1

        # now we can truncate the candidates
        candidates = candidates[:num_to_keep]
        # randomly shuffle the candidates so the top scoring ones are not always first
        self.rng.shuffle(candidates)
        # reassign the "id" field to each candidate after shuffling
        for i, candidate in enumerate(candidates):
            candidate["id"] = f"[{i + 1}]"

        # ordered dict preserves the key order
        candidates_dicts = [
            OrderedDict((k, candidate[k]) for k in self.fields_to_keep)
            for candidate in candidates
        ]
        # actual string to output
        candidates_string = "\n".join(
            [
                json.dumps(candidate_dict, indent=2)
                for candidate_dict in candidates_dicts
            ]
        )

        # ground truth of rank ordering is based on the score field
        ranking_tuples = [
            (candidate["id"], candidate["score"]) for candidate in candidates
        ]
        # sort by score
        ranking_tuples = sorted(ranking_tuples, key=lambda x: x[1], reverse=True)
        # now output the correct ranking string with = as ties and > as non-ties
        ranking_string = ""
        for i, (id, score) in enumerate(ranking_tuples):
            if i > 0:
                prev_score = ranking_tuples[i - 1][1]
                if score == prev_score:
                    ranking_string += " = "
                else:
                    ranking_string += " > "
            ranking_string += id

        instance_out = {
            "_id": instance["_id"],
            "query": instance["query"],
            "papers": candidates_string,
            "ranking": ranking_string,
        }
        return instance_out

    def format_instances(self, dataset):
        instances = []
        for i, instance in enumerate(dataset):
            instance["_id"] = i
            instances.append(self.format_instance(instance))

        return instances


# quick test
if __name__ == "__main__":
    import datasets

    instances = datasets.load_dataset(
        "allenai/scirepeval", "trec_covid", "evaluation", trust_remote_code=True
    )
    dataset = instances["evaluation"]
    formatter = Formatter(fields_to_keep=["title", "abstract", "date"])
    formatted = formatter.format_instances(dataset)

    # how how many candidates in each instance?
    # count the '{' in each instance
    for instance in formatted:
        print(instance["papers"].count("{"), end=" ")
