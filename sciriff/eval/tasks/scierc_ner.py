import json

from sciriff.eval.metrics.ner_f1 import NERF1
from ._base import JSONTask


class SciERC(JSONTask):
    default = {
        "Task": [],
        "Method": [],
        "Metric": [],
        "Material": [],
        "OtherScientificTerm": [],
        "Generic": [],
    }

    @staticmethod
    def make_flattened_metrics(res):
        """
        make flattened version of metrics for table display.
        """
        metrics_flat = {}
        for setting in ["valid_json", "all"]:
            for method in ["typed", "untyped"]:
                for k in ["p", "r", "f1"]:
                    keyname = f"{setting}_{method}_{k}"
                    metrics_flat[keyname] = res["f1"][setting][method][k]
        metrics_flat["bleu"] = res["bleu"]
        metrics_flat["json_parsed"] = res["json_counts"]["frac_success"]

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        metrics_summary = {"f1": res["f1"]["all"]["typed"]["f1"]}
        return metrics_summary

    def evaluate(self):
        res = {}
        predictions, json_counts = self.parse_predictions()

        evaluator = NERF1()
        res["f1"] = {}
        res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
        res["f1"]["all"] = evaluator.evaluate(predictions["all"])
        res["bleu"] = self.get_bleu()
        res["json_counts"] = json_counts

        self.dump_results(res)
