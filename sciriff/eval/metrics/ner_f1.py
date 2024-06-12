from collections import Counter

from sciriff.eval.metrics import util


class NERF1:
    "Exact-match NER F1."
    @staticmethod
    def _normalize(xs):
        "Normalize a single predicted list by converting to lowercase strings."
        return [util.normalize_list_entry(x) for x in xs]

    @staticmethod
    def _dict_to_tuples(ent_dict):
        """
        Models predict a dict whose keys are entity types and values are mentions.
        Convert to a list of (mention, type)  tuples.
        """
        res_typed = set()
        res_untyped = set()

        for ent_type, mentions in ent_dict.items():
            # Sometimes the model outputs a nested list of mentions by mistake; just
            # flatten these.
            mentions = util.flatten(mentions)
            for mention in mentions:
                mention_normalized = util.normalize_list_entry(mention)
                res_typed.add((mention_normalized, ent_type))
                res_untyped.add(mention_normalized)

        return res_typed, res_untyped

    def _evaluate_one(self, instance):
        pred_typed, pred_untyped = self._dict_to_tuples(instance["pred"])
        ref_typed, ref_untyped = self._dict_to_tuples(instance["ref"])

        self.counts_typed["correct"] += len(pred_typed & ref_typed)
        self.counts_typed["refs"] += len(ref_typed)
        self.counts_typed["preds"] += len(pred_typed)

        self.counts_untyped["correct"] += len(pred_untyped & ref_untyped)
        self.counts_untyped["refs"] += len(ref_untyped)
        self.counts_untyped["preds"] += len(pred_untyped)

    def evaluate(self, instances):
        self.counts_typed = util.count_dict(["correct", "preds", "refs"])
        self.counts_untyped = util.count_dict(["correct", "preds", "refs"])

        for instance in instances:
            self._evaluate_one(instance)

        res = {}
        res["typed"] = util.compute_f1(
            self.counts_typed["correct"], self.counts_typed["preds"], self.counts_typed["refs"]
        )
        res["untyped"] = util.compute_f1(
            self.counts_untyped["correct"],
            self.counts_untyped["preds"],
            self.counts_untyped["refs"],
        )

        return res
