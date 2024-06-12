from ._base import JSONTask
from sciriff.eval.metrics.list_f1 import ListF1

# NOTE(davidw) A lot of these fail for the base tulu model; this is always because the
# model `</s>`, which is pretty much the same as predicting no categories.


class MultiCite(JSONTask):
    default = []

    @staticmethod
    def make_flattened_metrics(res):
        """
        Make flattened version of metrics for table display.
        """
        metrics_flat = {}
        for setting in ["valid_json", "all"]:
            for k in ["p", "r", "f1"]:
                keyname = f"{setting}_{k}"
                metrics_flat[keyname] = res["f1"][setting]["f1"][k]
            metrics_flat[f"{setting}_frac_parsed"] = res["f1"][setting]["parse_counts"][
                "frac_success"
            ]
        metrics_flat["bleu"] = res["bleu"]
        metrics_flat["json_parsed"] = res["json_counts"]["frac_success"]

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        metrics_summary = {"f1": res["f1"]["all"]["f1"]["f1"]}
        return metrics_summary


    def evaluate(self):
        res = {}
        predictions, json_counts = self.parse_predictions()

        evaluator = ListF1()
        res["f1"] = {}
        res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
        res["f1"]["all"] = evaluator.evaluate(predictions["all"])
        res["bleu"] = self.get_bleu()
        res["json_counts"] = json_counts

        self.dump_results(res)
