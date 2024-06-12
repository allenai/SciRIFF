from sciriff.eval.metrics.relation_f1 import RelationF1
from ._base import JSONTask


class EvidenceInference(JSONTask):
    default = []

    @staticmethod
    def make_flattened_metrics(res):
        """
        Make flattened version of metrics for table display.
        """
        metrics_flat = {}
        for setting in ["valid_json", "all"]:
            metrics_flat[f"{setting}_frac_valid_list"] = res["f1"][setting][
                "parse_counts"
            ]["frac_success"]
            for method in ["exact", "substring", "overlap"]:
                for k in ["p", "r", "f1"]:
                    keyname = f"{setting}_{method}_{k}"
                    metrics_flat[keyname] = res["f1"][setting][method][k]
        metrics_flat["bleu"] = res["bleu"]
        metrics_flat["json_parsed"] = res["json_counts"]["frac_success"]

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        metrics_summary = {
            f"f1_{method}": res["f1"]["all"][method]["f1"]
            for method in ["exact", "substring", "overlap"]
        }
        return metrics_summary

    def evaluate(self):
        res = {}
        predictions, json_counts = self.parse_predictions()

        evaluator = RelationF1(tup_len=5, overlap_thresh=0.3)
        res["f1"] = {}
        res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
        res["f1"]["all"] = evaluator.evaluate(predictions["all"])
        res["bleu"] = self.get_bleu()
        res["json_counts"] = json_counts

        self.dump_results(res)
