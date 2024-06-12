# Evaluation

This doc has more details on how the evaluation code is organized. For high-level information see, the main [README](../README.md)

## Making predictions

The [predict_eleuther.py](../script/eval/predict_eleuther.py) uses Eleuther to run predictions. The way Eleuther works for prediction is: point the evaluator at a directory, and give it a template specifying how to format each instance. Tulu-family models require inputs formatted like `<|user|>\n{{ input }}\n<|assistant|>`.

The configs to specify how evaluation is performed are located in [eleuther_templates](../sciriff/eval/eleuther_templates/). There's one directory for evaluation of Tulu models using the appropriate chat templtae (`tulu`), and one general template that works well for API models like GPT (`general`). We also used this `general` template to evaluate Llama 2 and Llama 3 in the paper; it's possible performance could be improved by adding Llama-specific templates.

The `predict_eleuther.py` script has a flag for `--tasks`. This should be set to `science_adapt` to run evals for all tasks, or can be set to a comma-separated list of tasks to evaluate. If left blank, it will evaluate on all tasks, but it will re-invoke the Eleuther harness each time and re-load the model for each new task; if invoked with `--tasks science_adapt`, the model is only loaded once.

## Evaluating

The evaluation script [compute_science_metrics.py](../sciriff/eval/script/compute_science_metrics.py) takes a directory with a collection of model predictions as input and computes metrics for all the predictions.

### Eval implementation

The evaluations for each task are implemented in [eval/tasks](../sciriff/eval/tasks/). Roughly, there are two types of tasks: those that expect a `json` output and those that don't. The ones with the `json` require some extra handling to try to extract the `json` output, keep track of how many parse failures there were, supply a default value in case of failures, and compute metrics over (1) all the instances, replacing failures with default values, and (2) just the instances that parsed successfully. This is all handled in the `JSONTask` class in [_base.py](../sciriff/eval/tasks/_base.py).

Individual task evaluations inherit from one of the classes in `_base.py`. For instance, [biored_ner](../sciriff/eval/tasks/biored_ner.py) inherits from `JSONTask` and runs json parsing before doing anything else.

After the initial preprocessing is done, each `task` calls out to a metric (or metrics) to compute results. The metrics are implemented in [eval/metrics](../sciriff/eval/metrics/). There's one file per type of metric; for instance, [ner_f1](../sciriff/eval/metrics/ner_f1.py) computes F1 scores for NER tasks. This metric is consumed by the [biored_ner](../sciriff/eval/tasks/biored_ner.py) task. After the metrics have been computed, `biored_ner` collects allthe results together and adds some extra information -- for instance, on the `json` parse success rate -- and dumps results to `metrics.json` in the `biored_ner` subfolder of the model prediction directory.

Once all metrics have been computed and stored, the evaluation script collects them all and puts them in spreadsheets for easier inspection.
