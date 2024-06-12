"""
Takes the outputs of Eleuther predictions and computes metrics on top of them.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
import re
import shutil

from sciriff.lib import paths
from sciriff.eval import tasks as eval_tasks
from sciriff.eval.tasks._base import EvalTask


def sort_key(name):
    name_no_seed = re.sub("-seed.*", "", name)
    splt = name_no_seed.split("_")
    prefix = "_".join(splt[:-1])
    try:
        n_insts = int(splt[-1])
    except ValueError:
        n_insts = 0

    return (prefix, n_insts)


class ScienceEvaluator:
    "Runs all science evals and dumps results in a table."

    def __init__(self, args):
        self.pred_dir = Path(args.pred_dir)
        self.metrics_dir = Path(args.metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        self.metrics_by_model_dir = self.metrics_dir / "by_model"
        self.metrics_by_model_dir.mkdir(exist_ok=True)
        self.metrics_table_dir = self.metrics_dir / "tables"
        self.metrics_table_dir.mkdir(exist_ok=True)
        self.max_instances = args.max_instances
        self.clobber = args.clobber
        self.model_pattern = (
            re.compile(args.model_pattern) if args.model_pattern is not None else None
        )
        self.model_name = args.model_name

    def make_summary_table(self, metrics_summary):
        "Make summary table of all results."
        # List of metrics to use to compute a single science score.
        summary_metrics = [
            ("bioasq", "f1"),
            ("biored", "f1"),
            ("discomat", "bleu"),
            ("evidence_inference", "f1_overlap"),
            ("multicite", "f1"),
            ("mup", "lm_judge_reference"),
            ("qasper", "lm_judge_answer"),
            ("qasper", "f1_evidence"),
            ("scierc", "f1"),
            ("scifact", "f1_label"),
            ("scifact", "f1_evidence_token"),
        ]

        tbl = pd.DataFrame(metrics_summary)
        pvt = tbl.pivot_table(index="model", columns=["task", "metric"], values="value")

        # Remove columns if any are missing.
        summary_cols = [col for col in summary_metrics if col in list(pvt.columns)]
        if summary_cols != summary_metrics:
            missing = [col for col in summary_metrics if col not in list(pvt.columns)]
            print(f"The following expected metrics are missing: {missing}.")
        pvt = pvt[summary_cols]
        the_mean = pvt[summary_cols].mean(axis=1)
        the_median = pvt[summary_cols].median(axis=1)
        pvt["mean"] = the_mean
        pvt["median"] = the_median

        pvt.index.name = None
        ix_sorted = sorted(pvt.index, key=sort_key)
        pvt = pvt.loc[ix_sorted] * 100
        pvt.to_csv(
            self.metrics_table_dir / "summary.tsv", sep="\t", float_format="%0.2f"
        )

    def make_task_tables(self, metrics_flat):
        for task_name, metrics_task in metrics_flat.items():
            df = pd.DataFrame(metrics_task).set_index("model")
            ix_sorted = sorted(df.index, key=sort_key)
            df = df.loc[ix_sorted]
            df.to_csv(
                self.metrics_table_dir / f"{task_name}.tsv",
                sep="\t",
                float_format="%0.2f",
            )

    def reorganize_predictions(self, model_dir):
        "Put predictions for each task in its own directory."

        def get_task_name(pred_file, eval_tasks):
            eval_task = [task for task in eval_tasks if task in pred_file.name]
            if len(eval_task) != 1:
                raise ValueError("Couldn't find a matching eval task.")
            return eval_task[0]

        old_dir = model_dir / "science_adapt"

        # Get list of eval tasks.
        eval_task_dir = paths.EVAL_DIR / "eleuther_templates/tulu"
        names = [p.stem for p in eval_task_dir.glob("*.yaml")]
        eval_tasks = [name for name in names if name != "_default_template"]

        pred_files = [
            f
            for f in old_dir.iterdir()
            if "pretrained__" in f.name or "model__" in f.name
        ]
        eleuther_file = old_dir / "eleuther.jsonl"
        for pred_file in pred_files:
            task_name = get_task_name(pred_file, eval_tasks)
            new_dir = model_dir / task_name
            new_dir.mkdir(exist_ok=True)
            pred_file.rename(new_dir / pred_file.name)
            shutil.copyfile(eleuther_file, new_dir / "eleuther.jsonl")

        eleuther_file.unlink()
        old_dir.rmdir()

    def eval_one_model(self, model_dir):
        model_name = model_dir.name

        if [x.name for x in model_dir.iterdir()] == ["science_adapt"]:
            # Re-organize the predictions so that there's 1 folder per task.
            self.reorganize_predictions(model_dir)

        for pred_dir in model_dir.iterdir():
            if not [fname for fname in pred_dir.iterdir()]:
                # Skip directories with no predictions
                continue

            task_name = pred_dir.name
            print(task_name)
            eval_dir = self.metrics_by_model_dir / model_name / task_name
            eval_dir.mkdir(exist_ok=True, parents=True)

            # Dump evaluations using this tasks's evaluator.
            # NOTE `baseline_dir` used to be the directory of a model to compare against
            # as a baseline for GPT-judge evals; reference-based evals seem fine so we
            # don't compare against a baseline model anymore.
            evaluator = eval_tasks.registry.get(task_name, EvalTask)(
                pred_dir, eval_dir, baseline_dir=None, max_instances=self.max_instances
            )

            # If there's already a metrics file, skip unless `--clobber` is given.
            metrics_flat_file = evaluator.eval_dir / "metrics_flat.json"
            if (not metrics_flat_file.exists()) or self.clobber:
                evaluator.evaluate()

            # Add to spreadsheet of metrics for this task specifically.
            metrics_flat = json.load(open(metrics_flat_file))
            metrics_flat["model"] = model_dir.name
            if task_name in self.metrics_flat:
                self.metrics_flat[task_name].append(metrics_flat)
            else:
                self.metrics_flat[task_name] = [metrics_flat]

            # Add to summary spreadsheet.
            metrics_summary = json.load(
                open(evaluator.eval_dir / "metrics_summary.json")
            )
            for k, v in metrics_summary.items():
                # Shorten task name for readability.
                task_alias = (
                    task_name.split("_")[0]
                    if task_name != "evidence_inference"
                    else task_name
                )
                to_append = {
                    "model": model_dir.name,
                    "task": task_alias,
                    "metric": k,
                    "value": v,
                }
                self.metrics_summary.append(to_append)

    def run_eval(self):
        self.metrics_summary = []
        self.metrics_flat = {}

        # Evaluate only a single model if requested.
        if self.model_name is not None:
            model_dir = self.pred_dir / self.model_name
            self.eval_one_model(model_dir)
            return

        # Otherwise, run evals for each task and model and dump results.
        for model_dir in self.pred_dir.iterdir():
            # Skip models that don't match pattern, if pattern requested.
            if self.model_pattern is not None:
                if not self.model_pattern.match(model_dir.name):
                    continue

            print(model_dir.name)
            self.eval_one_model(model_dir)

        self.make_task_tables(self.metrics_flat)
        self.make_summary_table(self.metrics_summary)


def make_parser():
    parser = ArgumentParser(
        description="""Given a `pred_dir` directory holding predictions from
        different models, loop over all the model directories and evaluate all
        predictions."""
    )
    parser.add_argument(
        "--pred_dir", type=str, help="Directory holding predictions for all models."
    )
    parser.add_argument(
        "--metrics_dir", type=str, help="Directory where metrics should go."
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        help="If given, only evaluate this many instances per task. Helps for debugging.",
        default=None,
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="If given, re-compute metrics even if result files are already in place.",
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        help="If given, only evaluate models whose names match this pattern.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="If given, only evaluate the model with this name and don't make summary table.",
        default=None,
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    se = ScienceEvaluator(args)
    se.run_eval()


if __name__ == "__main__":
    main()
