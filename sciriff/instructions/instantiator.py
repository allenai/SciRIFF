"""
Instantiates templates for a set of tasks.
"""

import time
from transformers import LlamaTokenizerFast
from datasets import Dataset
from pathlib import Path
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from sciriff.instructions.task import TaskCollection
from sciriff.lib import paths


class Instantiator:
    "Runs template instantiation."

    def __init__(self, args, tokenizer="meta-llama/Llama-2-7b-hf"):
        self.task_collection = TaskCollection()
        self.args = args
        if args.out_dir is None:
            self.out_dir = paths.INSTANCE_DIR
        else:
            self.out_dir = Path(args.out_dir)

        # Frations to split off for validation or test data.
        self.split_frac = 0.2

        # Load the Llama2 tokenizer; we'll use this to get token counts for each instance.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer)

        # Tulu adds some special tokens; to be safe keep the max length to
        # `context_window - pad_tokens`.
        self.pad_tokens = 20

    def safe_instantiate_task(self, task_name):
        """
        Attempt to instantiate. If it works, return None; else return the error message.
        """
        try:
            self.instantiate_task(task_name)
            return None
        except Exception as e:
            error_info = (task_name, str(e))
            return error_info

    def instantiate_task(self, task_name):
        "Instantiate a single task."
        task = self.task_collection.tasks[task_name]
        task_dir = self.out_dir / f"{task_name}"

        if task_dir.exists() and any(task_dir.iterdir()) and self.args.no_clobber:
            return

        task_dir.mkdir(exist_ok=True, parents=True)
        splits = task.get_split_names()
        split_dict = {}
        for split in splits:
            # If necessary, map from canonical split name to HF names for this dataset.
            if task.split_lookup is None:
                hf_split_name = split
            else:
                hf_split_name = task.split_lookup[split]

            this_split = task.apply(
                split=hf_split_name,
                n_instances=self.args.n_instances,
                context_window=self.args.context_window - self.pad_tokens,
                template_ids=self.args.templates,
                random_template=self.args.random_template,
                tokenizer=self.tokenizer,
            )
            split_dict[split] = Dataset.from_list(this_split)

        res = self.standardize_splits(split_dict, task)
        self._check_no_overlap(res, task)

        for fold, ds in res.items():
            ds.to_json(f"{task_dir}/{fold}.jsonl")

    def standardize_splits(self, split_dict, task):
        """
        Some datasets don't have all three splits available. If this happens, rebalance
        to have all 3 splits.
        """
        # If we're not standardizing splits, for this task, just return.
        if not task.standardize_splits:
            return split_dict

        available_splits = set(split_dict)
        if available_splits == {"train", "validation", "test"}:
            # If we've got all splits, just return.
            return split_dict
        elif len(available_splits) == 2:
            if "train" not in available_splits:
                raise ValueError(f"{task.name} has val and test but no train.")

            # The evaluation split we have available. Use this as test, and split off
            # part of train as validation.
            eval_split = list(available_splits - {"train"})[0]
            n_validation = min(
                len(split_dict[eval_split]),
                round(len(split_dict["train"]) * self.split_frac),
            )
            new_split = split_dict["train"].train_test_split(
                test_size=n_validation, generator=task.rng
            )
            return {
                "train": new_split["train"],
                "validation": new_split["test"],
                "test": split_dict[eval_split],
            }
        elif len(set(split_dict)) == 1:
            # If there's only one split available, split it 3 ways.
            the_split = list(split_dict.keys())[0]
            the_data = split_dict[the_split]
            n_split = round(len(the_data) * self.split_frac)
            split_1 = the_data.train_test_split(test_size=n_split, generator=task.rng)
            split_2 = split_1["train"].train_test_split(
                test_size=n_split, generator=task.rng
            )
            return {
                "train": split_2["train"],
                "validation": split_2["test"],
                "test": split_1["test"],
            }
        else:
            raise ValueError(f"Unexpected splits found for {task.name}.")

    @staticmethod
    def _check_no_overlap(data_dict, task):
        "Double-check the final splits have no overlap in their instances."
        ids = {}
        for split, ds in data_dict.items():
            ids[split] = set(ds["_instance_id"])

        for split_1, split_2 in itertools.combinations(ids.keys(), 2):
            if ids[split_1] & ids[split_2]:
                raise ValueError(f"Overlap in instance ID's for task {task.name}.")

    def instantiate_no_catch_errors(self, task_names):
        "Instantiate and raise errors if they happen."
        for task_name in tqdm(task_names):
            self.instantiate_task(task_name)

    def instantiate_catch_errors(self, task_names):
        """Instantiate while catching errors; print error report at the end."""
        if self.args.workers:
            # If parallelizing, map over workers.
            print(f"Instantiating in parallel with {self.args.workers} workers.")
            with ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                res = executor.map(self.safe_instantiate_task, task_names)
        else:
            # Otherwise, loop and catch errors.
            res = []
            for task_name in tqdm(task_names):
                res.append(self.safe_instantiate_task(task_name))

        # Parse results and print stats.
        total_tasks = len(task_names)
        successes = []
        failures = []
        for entry in res:
            if entry is None:
                successes.append(entry)
            else:
                failures.append(entry)

        success_count = len(successes)
        print(f"Total of {success_count} / {total_tasks} instantiated.")

        for task_name, error_message in failures:
            print(f"Failed to instantiate {task_name}: {error_message}")

    def instantiate(self):
        "Instantiate all requested tasks."
        tick = time.time()

        self.out_dir.mkdir(exist_ok=True)

        all_tasks = list(self.task_collection.tasks.keys())
        if self.args.tasks is None:
            task_names = all_tasks
        else:
            task_names = self.args.tasks.split(",")
            for task_name in task_names:
                if task_name not in all_tasks:
                    raise ValueError(f"Unknown task name {task_name}.")

        if self.args.workers is None and not self.args.catch_errors:
            self.instantiate_no_catch_errors(task_names)
        else:
            self.instantiate_catch_errors(task_names)

        elapsed = (time.time() - tick) / 60
        msg = f"Elapsed time: {elapsed:.2f} minutes."
        print(msg)
