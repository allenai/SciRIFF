from collections import Counter

from sciriff.eval.metrics import util


class FactCheckingEval:
    "Fact-checking. Compute label and evidence F1."

    @staticmethod
    def _normalize(xs):
        "Normalize a single predicted list by converting to lowercase strings."
        return [util.normalize_list_entry(x) for x in xs]

    def _evaluate_one(self, instance):
        # Compute label F1.
        verdict_ref = instance["ref"]["verdict"]
        if not isinstance(instance["pred"], dict):
            self.failure_counts["not_dict"] += 1
            return

        verdict_pred = instance["pred"].get("verdict", None)
        if verdict_pred is None:
            self.failure_counts["correct_verdict"]["no"] += 1
            verdict_pred = "NEI"
        else:
            self.failure_counts["correct_verdict"]["yes"] += 1

        if verdict_ref != "NEI":
            self.label_counts["refs"] += 1
            if verdict_pred == verdict_ref:
                self.label_counts["correct"] += 1
        if verdict_pred != "NEI":
            self.label_counts["preds"] += 1

        # Compute sentence-level evidence F1.
        ev_ref = instance["ref"]["evidence"]
        ev_pred = instance["pred"].get("evidence", None)
        if ev_pred is None:
            self.failure_counts["correct_evidence"]["no"] += 1
            ev_pred = []
        else:
            self.failure_counts["correct_evidence"]["yes"] += 1

        ev_set_ref = set([sent.strip().lower() for sent in ev_ref])
        # Sometimes models will predict a nested list; just flatten it.
        ev_pred_flat = util.flatten(ev_pred)
        ev_set_pred = set([sent.strip().lower() for sent in ev_pred_flat])

        self.evidence_counts["correct"] += len(ev_set_ref & ev_set_pred)
        self.evidence_counts["preds"] += len(ev_set_pred)
        self.evidence_counts["refs"] += len(ev_set_ref)

        # Compute evidence token f1.
        ev_toks_ref = " ".join(ev_ref)
        ev_toks_pred = " ".join(ev_pred_flat)
        reference_tokens = util.token_f1_normalize_answer(ev_toks_ref).split()
        prediction_tokens = util.token_f1_normalize_answer(ev_toks_pred).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        self.token_counts["refs"] += len(reference_tokens)
        self.token_counts["preds"] += len(prediction_tokens)
        self.token_counts["correct"] += sum(common.values())

    def evaluate(self, instances):
        self.label_counts = util.count_dict(["refs", "preds", "correct"])
        self.evidence_counts = util.count_dict(["refs", "preds", "correct"])
        self.token_counts = util.count_dict(["refs", "preds", "correct"])
        self.failure_counts = {
            "correct_verdict": util.count_dict(["yes", "no"]),
            "correct_evidence": util.count_dict(["yes", "no"]),
            "not_dict": 0
        }

        for instance in instances:
            self._evaluate_one(instance)

        f1_label = util.compute_f1(
            self.label_counts["correct"],
            self.label_counts["preds"],
            self.label_counts["refs"],
        )
        f1_evidence_sent = util.compute_f1(
            self.evidence_counts["correct"],
            self.evidence_counts["preds"],
            self.evidence_counts["refs"],
        )
        f1_evidence_tok = util.compute_f1(
            self.token_counts["correct"],
            self.token_counts["preds"],
            self.token_counts["refs"],
        )

        # Get the fraction that were successful.
        for k in ["correct_verdict", "correct_evidence"]:
            self.failure_counts[k]["frac_success"] = util.safe_div(
                self.failure_counts[k]["yes"], util.sum_dict(self.failure_counts[k])
            )

        res = {
            "f1": {
                "label": f1_label,
                "evidence_sent": f1_evidence_sent,
                "evidence_tok": f1_evidence_tok,
            },
            "answer_parse": self.failure_counts,
        }

        return res
