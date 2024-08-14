"""
Code to do templating for a single task.
"""

import yaml
import datasets
import importlib
from typing import List, Dict, Optional
import warnings
from numpy import random as npr
import inspect

from sciriff.instructions.template import Template
from sciriff.lib import paths


class Task(yaml.YAMLObject):
    """
    Class with all templates for a single task. This correpsonds to a single yaml file.
    """

    yaml_tag = "!Task"

    @classmethod
    def from_yaml(cls, loader, node):
        "Fanciness to enable some values to be missing."
        # Load the node to a python dictionary
        values = loader.construct_mapping(node, deep=True)

        # Set default values for any missing fields
        values.setdefault("dataset", None)
        values.setdefault("subset", None)
        values.setdefault("data_dir", None)
        values.setdefault("formatter", None)
        values.setdefault("skip_splits", None)
        values.setdefault("evaluate", None)
        values.setdefault("split_lookup", None)
        values.setdefault("standardize_splits", True)

        return cls(**values)

    def __init__(
        self,
        name: str,
        metadata: dict,
        templates: List[dict],
        dataset: Optional[str] = None,
        subset: Optional[str] = None,
        data_dir: Optional[str] = None,
        formatter: Optional[dict] = None,
        skip_splits: Optional[List] = None,
        split_lookup: Optional[Dict] = None,
        standardize_splits: Optional[bool] = True,
        evaluate: Optional[dict] = None,
    ):
        self.name = name
        self.dataset = dataset
        self.subset = subset
        self.data_dir = data_dir
        self.formatter = formatter
        self.skip_splits = skip_splits
        self.split_lookup = split_lookup
        self.standardize_splits = standardize_splits
        self.metadata = metadata
        self.templates = {k: Template(**v) for k, v in templates.items()}
        self.evaluate = evaluate

        # Tell the templates their ID's.
        for template_id, template in self.templates.items():
            template._id = template_id

        # Give each task a unique random number generator; this will be useful for
        # random sampling that needs to happen.
        self.rng = npr.default_rng(seed=abs(hash(self.name)))

    @staticmethod
    def from_file(name):
        # Add a tag to the top level so the loader recognizes it.
        yaml_text = "!Task\n" + open(paths.TEMPLATE_DIR / f"{name}.yaml").read()
        return yaml.load(yaml_text, Loader=yaml.FullLoader)

    def get_split_names(self):
        allowed_splits = set(["train", "validation", "test"])
        if self.data_dir is not None:
            data_path = f"{paths.DATA_DIR}/preprocessing/processed/{self.data_dir}"
        else:
            data_path = self.dataset

        if self.subset is None:
            splits = datasets.get_dataset_split_names(data_path, trust_remote_code=True)
        else:
            splits = datasets.get_dataset_split_names(data_path, self.subset, trust_remote_code=True)

        # If a split has "non-standard" names, map to the standard names.
        if self.split_lookup is not None:
            reverse_lookup = {v: k for k, v in self.split_lookup.items()}
            splits = [reverse_lookup[split] for split in splits]

        # Get rid of any splits that we should skip for this dataset (e.g. because
        # outputs are blank).
        if self.skip_splits is not None:
            splits = [split for split in splits if split not in self.skip_splits]

        # Get rid of any invalid split names; warn if they appear.
        diff = set(splits) - allowed_splits
        if diff:
            msg = f"Unexpected splits {diff} for dataset {self.dataset}."
            warnings.warn(msg)

        splits = [split for split in splits if split in allowed_splits]

        return splits

    def get_template_ids(self):
        """Returns the available template IDs for this task."""
        return list(self.templates.keys())

    def apply(
        self,
        split="train",
        context_window=4096,
        n_instances=None,
        template_ids=None,
        random_template=False,
        truncate=False,
        highlight_variables=False,
        tokenizer=None,
    ):
        """
        Apply template to dataset.

        :param split: The split to format. Note that if split == "test", the
            `template_ids` and `random_template` flags are ignored and the `evaluate`
            template is used for all instance.
        :param n_instances: If not None, only apply to n_instances random instances.
        :param context_window: Max tokens per instance.
        :param template_ids: Only instantiate the templates in the given list.
        :param random_template: If given, instantiate each example with a random
            template selected from `template_ids`.
        :param truncate: Truncate the output.
        :param highlight_variables: highlight the added variables.
        :param tokenizer: If given, include intput and output token counts.
        :return: Dict with input and output.
        """
        if self.data_dir is not None:
            # If a data dir is specified, load from file.
            instances = datasets.load_dataset(
                "json",
                data_dir=f"{paths.DATA_DIR}/preprocessing/processed/{self.data_dir}",
                split=split,
            )
        else:
            # Otherwise load from Huggingface.
            instances = datasets.load_dataset(
                self.dataset, self.subset, split=split, trust_remote_code=True
            )

        # Assign indices into the original dataset instances.
        _ids = list(range(len(instances)))
        instances = instances.add_column("_id", _ids)
        # If the number of instances is capped, shuffle them.
        if n_instances is not None:
            instances = instances.shuffle(generator=self.rng)

        # Transformer instances if required.
        if self.formatter is not None:
            formatter_module = importlib.import_module(
                f"sciriff.instructions.formatters.{self.formatter['name']}"
            )
            formatter_args = self.formatter["args"]
            if formatter_args is None:
                formatter_args = {}
            # Add `context_window` if the formatter expects it. This is a hack.
            expected_args = [
                p.name
                for p in inspect.signature(
                    formatter_module.Formatter.__init__
                ).parameters.values()
            ]
            if "context_window" in expected_args:
                formatter_args["context_window"] = context_window

            formatter = formatter_module.Formatter(**formatter_args)
            instances = formatter.format_instances(instances)
            instances = datasets.Dataset.from_list(instances)

        # Get the selected templates.
        if split == "test":
            # Always use the first template for evaluation.
            template_ids = [0]
        elif template_ids is None:
            # Use all templates if none are specified.
            template_ids = sorted(self.templates.keys())
        res_metadata = self.metadata.copy()
        res_metadata["split"] = split  # Add the split info to the metadata
        res = []

        # Keep track of total instances, and stop if we hit limit (if given).
        instances_so_far = 0

        for instance in instances:
            if random_template:
                # Select a random template for each instance
                template_id = self.rng.choice(template_ids)
                task_templates = [self.templates[template_id]]
            else:
                # If not random_template, use selected or all templates
                task_templates = [self.templates[k] for k in template_ids]

            for template in task_templates:
                instance_id = f"{self.name}:{split}:{instance['_id']}"
                formatted_instance = template.apply(
                    instance_id, instance, res_metadata, truncate, highlight_variables
                )
                if tokenizer is not None:
                    formatted_instance["_input_toks"] = len(
                        tokenizer.tokenize(formatted_instance["input"])
                    )
                    formatted_instance["_output_toks"] = len(
                        tokenizer.tokenize(formatted_instance["output"])
                    )

                # Confirm that the instance doesn't have blank input or output.
                if formatted_instance["input"] == "":
                    raise ValueError(f"Blank input for {split} split of {self.name}.")
                if formatted_instance["output"] == "":
                    raise ValueError(f"Blank output for {split} split of {self.name}.")

                total_toks = (
                    formatted_instance["_input_toks"]
                    + formatted_instance["_output_toks"]
                )
                # Only keep instances that aren't too long.
                if total_toks <= context_window:
                    res.append(formatted_instance)
                    instances_so_far += 1
                    # If we've hit the instance cap, return here.
                    if n_instances is not None and instances_so_far >= n_instances:
                        return res

        return res


class TaskCollection:
    """
    Collects templates for all tasks.
    """

    def __init__(self):
        self.tasks = self._collect_tasks()

    def _collect_tasks(self):
        template_dict = {}
        for template_file in paths.TEMPLATE_DIR.iterdir():
            template_name = template_file.stem
            template_dict[template_name] = Task.from_file(template_name)

        return template_dict
