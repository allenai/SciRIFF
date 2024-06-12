"""
Base class to handle shared tasks like loading in the predictions.
"""

import json
from pathlib import Path
from sciriff.lib import util
import evaluate

from sciriff.eval.metrics.json_parser import JSONParser


class EvalTask:
    def __init__(self, pred_dir, eval_dir, baseline_dir, max_instances=None):
        self.pred_dir = Path(pred_dir)
        self.baseline_dir = baseline_dir
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(exist_ok=True, parents=True)
        self.max_instances = max_instances

        # Get the prediction file.
        self.pred_file = self.get_prediction_file(self.pred_dir)

    def get_prediction_file(self, wkdir):
        "Get file holding predictions."
        pred_file = [
            x
            for x in wkdir.iterdir()
            if "pretrained__" in x.name or "model__" in x.name
        ]
        if len(pred_file) != 1:
            raise Exception(
                f"Found multiple files in {wkdir} that could have predictions."
            )
        return pred_file[0]

    def load_predictions(self, fname=None):
        fname = self.pred_file if fname is None else fname
        preds = json.load(open(fname))
        if self.max_instances is not None:
            preds = preds[: self.max_instances]

        return preds

    def get_bleu(self):
        "Compute bleu scores for all tasks."
        raw_predictions = self.get_raw_predictions()
        bleu_scorer = evaluate.load("bleu")
        predictions = [entry["pred"] for entry in raw_predictions]
        references = [entry["ref"] for entry in raw_predictions]

        res = bleu_scorer.compute(predictions=predictions, references=references)
        return res["bleu"]

    @staticmethod
    def make_flattened_metrics(res):
        "Flatten metrics for tabular output. Overwritten by subclasses."
        return res

    @staticmethod
    def make_summary_metrics(res):
        "Save overally summary metrics. Overwritten by subclasses."
        return res

    def dump_results(self, res):
        "Dump results to file."
        with open(self.eval_dir / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)

        with open(self.eval_dir / "metrics_flat.json", "w") as f:
            json.dump(self.make_flattened_metrics(res), f, indent=2)

        with open(self.eval_dir / "metrics_summary.json", "w") as f:
            json.dump(self.make_summary_metrics(res), f, indent=2)

    def get_raw_predictions(self, fname=None):
        entries = self.load_predictions(fname)
        raw_predictions = []
        for entry in entries:
            prompt = entry["arguments"][0][0]
            ref = entry["target"]
            # Tulu models usually end with `</s>`; strip it off.
            pred = entry["filtered_resps"][0].strip("</s>")
            raw_predictions.append({"prompt": prompt, "pred": pred, "ref": ref})

        return raw_predictions

    def evaluate(self):
        "Default eval just grabs BLEU scores and dumps raw predictions."
        # Always compute and save the bleu score.
        res = {"bleu": self.get_bleu()}

        # Save metrics (in this case just bleu).
        self.dump_results(res)

        # Save raw predictions to file.
        raw_predictions = self.get_raw_predictions()

        raw_file = self.eval_dir / "raw_predictions.jsonl"
        util.write_jsonl(raw_predictions, raw_file)


class JSONTask(EvalTask):
    "Base class for all tasks that have an initial json parsing step."
    default = None  # Overwritten by child classes.

    def __init__(self, pred_dir, eval_dir, baseline_dir, max_instances=None):
        super().__init__(pred_dir, eval_dir, baseline_dir, max_instances)
        self.json_parser = JSONParser(
            default=self.default, task=self.__class__.__name__
        )
        self.counts = {}

    @staticmethod
    def _make_pairs(preds, refs, prompts):
        """
        Make a list where each entry has a prediction and corresponding reference.
        Return two versions: one with all pairs including json parse failures, and one
        with only the pairs where the prediction parsed correctly.
        """
        all_pairs = []
        successful_pairs = []
        raw_pairs = []
        assert len(preds) == len(refs) == len(prompts)
        for pred, ref, prompt in zip(preds, refs, prompts):
            # Return the raw pairs so we can debug json parse failures.
            raw_pairs.append({"pred": pred, "ref": ref, "prompt": prompt})
            to_append = {"pred": pred["value"], "ref": ref["value"], "prompt": prompt}
            all_pairs.append(to_append)
            if pred["status"] != "extract_failure":
                successful_pairs.append(to_append)

        return {"parsed": successful_pairs, "all": all_pairs, "raw": raw_pairs}

    def parse_predictions(self):
        "Parse the predictions to json and keep trakc of how many failures there were."
        # Dump raw predictions to file.
        raw_file = self.eval_dir / "raw_predictions.jsonl"
        util.write_jsonl(self.get_raw_predictions(), raw_file)

        entries = self.get_raw_predictions()

        # Extract references and predictions as json.
        refs, counts_ref = self.json_parser([x["ref"] for x in entries])
        if counts_ref["valid_json"] != len(refs):
            raise Exception("References should always parse to json.")
        self.counts["json_ref"] = counts_ref

        preds, counts_pred = self.json_parser([x["pred"] for x in entries])
        self.counts["json_pred"] = counts_pred

        # Return versions of the data with and without parse failures.
        prompts = [x["prompt"] for x in entries]
        pairs = self._make_pairs(preds, refs, prompts)

        return pairs, self.counts["json_pred"]
