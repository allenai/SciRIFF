import pandas as pd
# from bert_score import BERTScorer
import numpy as np
import random
import torch
from transformers import pipeline

BERT_SCORE_TYPES = {"bertscore_p": 0, "bertscore_r": 1, "bertscore_f": 2}
ENTRY_TYPES = ["intervention", "comparator", "outcome", "effect", "evidence"]


class EvidenceInferenceEval:

    def __init__(self, threshold, model_name, score_type, n_samples=None):
        """
        :param threshold: a threshold to determine matching - each pair of prediction, reference who's score pass this threshold will be considered a match.
        :param model_name: name of model to be used as a scorer.
        :param n_samples: set this parameter to an integer representing sample size in order to only calculate scores over a random sample. default is None (no sampling, calculating using the entire dataset provided).
        :param score_type: type of score to calculate - choose between "bertscore_f","bertscore_r","bertscore_p", "nli".
        """
        self.n_samples = n_samples
        self.threshold = threshold
        if score_type not in ["bertscore_p", "bertscore_f", "bertscore_r", "nli"]:
            raise ValueError(
                "incompatible score_type - must be 'bertscore_<p/r/f>' or 'nli'"
            )
        if score_type == "nli" and "nli" not in model_name:
            raise ValueError(
                "incompatible model_name - must be an nli model if score_type='nli'"
            )
        self.model_name = model_name
        self.lang = "en-sci" if "sci" in model_name else "en"
        self.scores = {"by_entry": {}, "overall": {}}
        self.score_type = score_type
        self.bertscore_type = (
            BERT_SCORE_TYPES[score_type] if "bertscore" in score_type else None
        )

    @staticmethod
    def calculate_instance_scores(
        pred_matches, ref_matches, total_recall, total_precision, total_f1
    ):
        precision = (
            sum(pred_matches) / len(pred_matches) if len(pred_matches) > 0 else 0
        )  # ratio of predictions that had at least 1 matching reference
        recall = (
            sum(ref_matches) / len(ref_matches) if len(ref_matches) > 0 else 0
        )  # ratio of references that were matched to at least 1 prediction
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        total_recall.append(recall)
        total_precision.append(precision)
        total_f1.append(f1)

    @staticmethod
    def format_tuples_as_sentences(instance):
        predictions = [
            f"Intervention: '{e[0]}', Comparator: '{e[1]}', Outcome: '{e[2]}', Effect: '{e[3]}', Evidence: '{e[4]}'"
            for e in instance["prediction"]
            if len(e) == 5
        ]
        references = [
            f"Intervention: '{e[0]}', Comparator: '{e[1]}', Outcome: '{e[2]}', Effect: '{e[3]}', Evidence: '{e[4]}'"
            for e in instance["reference"]
            if len(e) == 5
        ]
        return predictions, references

    def get_score(self, instance_scores, pred, ref, scorer):
        if (pred, ref) in instance_scores:
            score = instance_scores[(pred, ref)]
        else:
            if "bertscore" in self.score_type:
                score = scorer.score([pred], [ref])[self.bertscore_type].item()
            else:
                score = scorer(f"[CLS] {ref} [SEP] {pred} [SEP]")[0]
            instance_scores[(pred, ref)] = score
        return score

    def log_entry_scores(self, entry_type, total_recall, total_precision, total_f1):
        self.scores["by_entry"][entry_type] = {
            "recall": np.mean(total_recall),
            "precision": np.mean(total_precision),
            "f1": np.mean(total_f1),
        }

    def log_overall_scores(self):
        self.scores["overall"]["per_entry"] = {
            "recall": np.mean(
                [item["recall"] for item in self.scores["by_entry"].values()]
            ),
            "precision": np.mean(
                [item["precision"] for item in self.scores["by_entry"].values()]
            ),
            "f1": np.mean([item["f1"] for item in self.scores["by_entry"].values()]),
        }

    def evaluate_by_entry_type(self, instances, scorer, save_individual_scores):
        """
        Calculates score by comparing pairs belonging to the same tuple entry (e.g. "Intervention").
        For each entry, for each instance, each combination of prediction[entry] and reference[entry]
        Then, each combination of prediction, reference sentences are evaluated using the provided scorer
        to determine if the pair is matching. The matches are used to calculate recall, precision and f1 per instance,
        which are averaged to get recall, precision and f1 for the entry over the entire dataset.
        This function also calculates overall scores, which averages each score per entry over all 5 entries.
        """
        if save_individual_scores:
            scores_per_entry = pd.DataFrame(
                columns=["instance", "entry_type", "prediction", "reference", "score"]
            )
        for i, entry_type in enumerate(ENTRY_TYPES):  # for each entry type
            total_recall, total_precision, total_f1 = [], [], []
            for j, instance in instances.items():
                instance_scores = {}
                if len(instance["prediction"]) == 0 or len(instance["reference"]) == 0:
                    continue
                pred_matches, ref_matches = [0] * len(instance["prediction"]), [
                    0
                ] * len(instance["reference"])

                # check matches between every possible combination of prediction & reference (within same instance and entry)
                for p, pred in enumerate(instance["prediction"]):
                    for r, ref in enumerate(instance["reference"]):
                        if (
                            len(pred) != 5
                            or len(ref) != 5
                            or pred[i] is None
                            or ref[i] is None
                        ):
                            continue

                        score = self.get_score(instance_scores, pred[i], ref[i], scorer)
                        if save_individual_scores:
                            new_row = pd.DataFrame(
                                [(j, entry_type, pred[i], ref[i], score)],
                                columns=scores_per_entry.columns,
                            )
                            scores_per_entry = pd.concat(
                                [scores_per_entry, new_row], ignore_index=True
                            )
                        if (
                            "bertscore" in self.score_type and score >= self.threshold
                        ) or (
                            "nli" in self.score_type
                            and score["label"] == "ENTAILMENT"
                            and score["score"] >= self.threshold
                        ):
                            pred_matches[p], ref_matches[r] = 1, 1
                self.calculate_instance_scores(
                    pred_matches, ref_matches, total_recall, total_precision, total_f1
                )
            self.log_entry_scores(entry_type, total_recall, total_precision, total_f1)
        self.log_overall_scores()

        if save_individual_scores:
            scores_per_entry.drop_duplicates(
                subset=["prediction", "reference"], inplace=True
            )
            scores_per_entry.sort_values(by="score", ascending=False, inplace=True)
            return scores_per_entry

    def evaluate_by_full_tuple(self, instances, scorer):
        """
        Calculates score by comparing the entire 5-tuple as a whole.
        For each instance, each prediction and reference 5-tuple is converted to a sentence.
        Then, each combination of prediction, reference sentences is evaluated using the provided scorer
        to determine if the pair is matching. The matches are used to calculate recall, precision and f1 per instance,
        which are averaged to get recall, precision and f1 in the full-tuple level on the entire dataset.
        """
        total_recall, total_precision, total_f1 = [], [], []
        for instance in instances.values():
            if len(instance["prediction"]) == 0 or len(instance["reference"]) == 0:
                continue
            predictions, references = self.format_tuples_as_sentences(instance)
            pred_matches, ref_matches = [0] * len(predictions), [0] * len(references)
            for p, pred in enumerate(predictions):
                for r, ref in enumerate(references):
                    if "bertscore" in self.score_type:
                        score = scorer.score([pred], [ref])[self.bertscore_type].item()
                    else:
                        score = scorer(f"[CLS] {ref} [SEP] {pred} [SEP]")[0]
                    if ("bertscore" in self.score_type and score >= self.threshold) or (
                        "nli" in self.score_type
                        and score["label"] == "ENTAILMENT"
                        and score["score"] >= self.threshold
                    ):
                        pred_matches[p], ref_matches[r] = 1, 1
            self.calculate_instance_scores(
                pred_matches, ref_matches, total_recall, total_precision, total_f1
            )
        self.scores["overall"]["per_full_tuple"] = {
            "recall": np.mean(total_recall),
            "precision": np.mean(total_precision),
            "f1": np.mean(total_f1),
        }

    def evaluate(
        self, instances, evaluate_by_full_tuple=True, individual_score_file=None
    ):
        """
        :param instances: a dictionary where keys are indices and values are a dictionaries with "prediction", "reference" keys and a list of tuples as values.
        :param evaluate_by_full_tuple: set to True to calculate overall scores not only per entry, but also on the tuple as a whole. default is True.
        :param individual_score_file: supply a path to a csv file save not only aggregated scores but also individual scores per evaluated pair.
        This could be used to manually evaluate the choice of model and threshold. default is None (not saving individual scores).
        """
        # sample examples if required
        if self.n_samples is not None:
            random.seed(42)
            chosen_samples = sorted(
                random.sample(range(len(instances)), self.n_samples)
            )
            instances = {
                key: value for key, value in instances.items() if key in chosen_samples
            }

        # Load model (for BERTscore / NLI)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "bertscore" in self.score_type:
            scorer = BERTScorer(
                model_type=self.model_name, lang=self.lang, device=device
            )
        else:
            scorer = pipeline("text-classification", model=self.model_name)

        # Calculate Scores
        scores_per_entry = self.evaluate_by_entry_type(
            instances, scorer, individual_score_file
        )
        if evaluate_by_full_tuple:
            self.evaluate_by_full_tuple(instances, scorer)

        if individual_score_file is not None:
            # Dump scores per pair per entry, for threshold analysis purposes
            scores_per_entry.to_csv(individual_score_file, index=False)

        return self.scores
