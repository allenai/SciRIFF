from ._base import EvalTask
from sciriff.eval.metrics.summary_comparison import SummaryComparison
import evaluate
from sciriff.lib import util
import json
from copy import deepcopy


class MUP(EvalTask):
    @staticmethod
    def make_flattened_metrics(res):
        metrics_flat = {}
        metrics_flat["bleu"] = res["bleu"]
        metrics_flat.update(res["rouge"])
        # Get all the lm judge metrics.
        for k, v in res["lm_judge"].items():
            for sub_k, sub_v in v.items():
                this_key = f"lm_judge_{k}_{sub_k.replace('lm_judge_', '')}"
                metrics_flat[this_key] = sub_v

        return metrics_flat

    @staticmethod
    def make_summary_metrics(res):
        return {
            "lm_judge_reference": res["lm_judge"]["reference_comparison"]["avg_rating"],
            "rouge": res["rouge"]["rougeL"],
        }

    def _initiate_instances(self):
        # NOTE We used to do a comparison using an LM judge to compare each model
        # against a baseline. Reference-based comparisons are easier and work fine, so
        # we don't do the head-to-head comparisons now.
        raw_predictions = self.get_raw_predictions(fname=self.pred_file)
        # raw_baselines = self.get_raw_predictions(fname=self.baseline_file)
        raw_file = self.eval_dir / "raw_predictions.jsonl"
        util.write_jsonl(raw_predictions, raw_file)

        predictions = [entry["pred"] for entry in raw_predictions]
        references = [entry["ref"] for entry in raw_predictions]
        # baselines = [entry["pred"] for entry in raw_baselines]
        prompts = [entry["prompt"] for entry in raw_predictions]

        # Make sure that the prompts match on the model and baseline.

        # wrap all the info for the llm judge into a dict
        instances = {}
        for i in range(len(predictions)):
            instances[i] = {}
            instances[i]["prediction"] = predictions[i]
            # instances[i]["baseline"] = baselines[i]
            instances[i]["prompt"] = prompts[i]
            instances[i]["reference"] = references[i]

        return instances, predictions, references

    def evaluate(self, use_batch_api):

        instances, predictions, references = self._initiate_instances()
        res = {}

        # Compute blue scores relative to gold summaries.
        res["bleu"] = self.get_bleu()

        # Compute rouge scores relative to gold summaries.
        rouge_scorer = evaluate.load("rouge")
        rouge = rouge_scorer.compute(predictions=predictions, references=references)
        res["rouge"] = rouge

        # use model judge to compare `predictions` against `baselines` and against gold standard summaries
        # Number of samples to evaluate for each task; use more for the ref comparison since it uses gpt-3.5.
        n_sample_lookup = {"model_comparison": 50,
                           "reference_comparison": 100}

        res["lm_judge"] = {}
        eval_type = "reference_comparison"

        # set output directories
        lm_judge_raw_results_file = self.eval_dir / f"lm_judge_{eval_type}_raw.json"
        lm_judge_agg_results_file = self.eval_dir / f"lm_judge_{eval_type}.json"
        self.lm_judge_file = lm_judge_agg_results_file
        lm_judge_outputs = self.eval_dir / f"lm_judge_raw.json"

        # check if eval output file exists, if not - run evaluation
        if self.lm_judge_file.exists():
            llm_judge_results = json.load(open(self.lm_judge_file))
            res["lm_judge"][eval_type] = llm_judge_results
            print("lm_judge_file found")
        else:
            evaluator = SummaryComparison()
            llm_judge_results = evaluator.evaluate(
                deepcopy(instances),
                lm_judge_raw_results_file,
                lm_judge_agg_results_file,
                eval_type,
                n_sample_lookup[eval_type],
                lm_judge_outputs,
                use_batch_api=use_batch_api
            )
            if not llm_judge_results:
                return
            res["lm_judge"][eval_type] = llm_judge_results

        self.dump_results(res)
