import json

from ._base import JSONTask
from sciriff.eval.metrics.fact_checking_eval import FactCheckingEval


class SciFact(JSONTask):
    default = {"verdict": "NEI", "evidence": []}

    @staticmethod
    def make_flattened_metrics(res):
        metrics_flat = {}
        for setting in ["valid_json", "all"]:
            # Get precision, recall, and f1 for all metrics.
            for method in ["label", "evidence_sent", "evidence_tok"]:
                for k in ["p", "r", "f1"]:
                    keyname = f"{setting}_{method}_{k}"
                    metrics_flat[keyname] = res["results"][setting]["f1"][method][k]
            # Get the fraction of times we could successfully parse the answers out of
            # the json result.
            for method in ["correct_verdict", "correct_evidence"]:
                metrics_flat[f"{setting}_{method}_frac_parsed"] = res["results"][
                    setting
                ]["answer_parse"][method]["frac_success"]
        metrics_flat["bleu"] = res["bleu"]
        metrics_flat["json_parsed"] = res["json_counts"]["frac_success"]

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        metrics_summary = {
            "f1_label": res["results"]["all"]["f1"]["label"]["f1"],
            "f1_evidence_sent": res["results"]["all"]["f1"]["evidence_sent"]["f1"],
            "f1_evidence_token": res["results"]["all"]["f1"]["evidence_tok"]["f1"],
        }
        return metrics_summary

    def evaluate(self):
        res = {}
        predictions, json_counts = self.parse_predictions()

        evaluator = FactCheckingEval()

        res["results"] = {}
        res["results"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
        res["results"]["all"] = evaluator.evaluate(predictions["all"])
        res["bleu"] = self.get_bleu()
        res["json_counts"] = json_counts

        self.dump_results(res)
