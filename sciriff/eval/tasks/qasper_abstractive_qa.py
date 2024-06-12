import json
import re

from ._base import JSONTask
from sciriff.eval.metrics.attributed_qa_eval import AttributedQAEval


class Qasper(JSONTask):
    default = {"answer": None, "evidence": None}

    @staticmethod
    def make_flattened_metrics(res):
        metrics_flat = {}
        for setting in ["valid_json", "all"]:
            # Get the fraction of times we could parse out answers.
            for method in ["answer", "evidence", "answer_entry_is_str"]:
                this_key = f"{setting}_{method}_frac_parsed"
                this_val = res["results"][setting]["answer_parse"][method][
                    "frac_success"
                ]
                metrics_flat[this_key] = this_val
            # Get precision, recall, and f1 for all metrics.
            for k, v in res["results"][setting]["scores"].items():
                keyname = f"{setting}_{k}"
                metrics_flat[keyname] = v

        metrics_flat["bleu"] = res["bleu"]
        metrics_flat["json_parsed"] = res["json_counts"]["frac_success"]

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        return {
            "lm_judge_answer": res["results"]["all"]["scores"].get("lm_judge", None),
            "f1_answer": res["results"]["all"]["scores"]["f1_answer_all"],
            "f1_evidence": res["results"]["all"]["scores"]["f1_evidence_all"],
        }

    def evaluate(self):
        res = {}
        predictions, json_counts = self.parse_predictions()

        lm_judge_file = self.eval_dir / "lm_judge.json"
        evaluator = AttributedQAEval()

        res["results"] = {}
        res["results"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
        res["results"]["all"] = evaluator.evaluate(
            predictions["all"], lm_judge_file=lm_judge_file
        )
        res["bleu"] = self.get_bleu()
        res["json_counts"] = json_counts

        self.dump_results(res)
