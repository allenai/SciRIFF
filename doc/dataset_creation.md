# Dataset creation

We use `yaml` files to specify how to transform a dataset into our instruction-following format. We use one `yaml` file per task; these are found in [sciriff/instructions/templates](../sciriff/instructions/templates). If you're interested in adding a new task, the steps are:

- Create a YAML file specifying the conversion process (details below).
- Run [script/instructions/validate.py](../script/instructions/validate.py) to validate the YAML file.
- Run [script/instructions/instantiate.py](../script/instructions/instantiate.py) to convert data into instruction instances as specified in the YAML. For more information, check out the arguments to `instantiate.py`.

Details on template creation are included below, organized as follows.

- [Template schema](#template-schema)
- [Task details](#task-details)
- [Template details](#template-details)

## Template schema

Our template format is similar to [promptsource](https://github.com/bigscience-workshop/promptsource), with a few modifications. The format is below. We use `?` in the schema to indicate that a field is optional and can be ommitted.

Unlike promptsource, a single yaml file should correspond to a _single task_ associated with a dataset. Multiple prompts associated with this task are welcome.

```yaml
name: string              # Your name for this task.
dataset?: string          # If available, the Huggingface dataset used for the task. If null, load from file instead.
subset?: string           # Optionally, the subset of the Huggingface dataset.
data_dir?: string         # If not on Huggingface, the data directory, specified relative to `data/processed`.
skip_splits?: list        # Some HF datasets have blind test sets with blank outputs; these should be skipped.
split_lookup?: dict       # If needed, mapping from canonical split names to names for this HF dataset.
standardize_splits?: bool # True by default. If false, don't re-balance the data to create 3 splits.
formatter?:               # Optionally, a formatter used to preprocess the dataset.
    name: string
    args:
        key1: value1
        key2: value2
        ...
metadata:                        # Task metadata
  task: string                   # The task type. For example, extractive question answering.
  domains: list[string]          # A list of domains for this dataset.
  source_type: string            # single doc or multiple?
  input_context: string          # The input context for this task.
  output_context: string         # The output context (e.g. a label, a paragraph).
  contributor: string            # An ID for the contributor (you).
templates:                       # The templates for this task.
  0:                   # Everything is the same as promptsource templates, except we
    jinja: string                # number the templates starting at 0 rather than using unique ID's.
    answer_choices?: string
  1:                             # Keep counting.
    ...
```

## Task details

- `name`: The `name` field should match the name of the `yaml` file containing it. It should have the form `<dataset name>_<task name>`. For instance, if your source dataset is Qasper and the task is abstractive QA, a good `name` would be `qasper_abstractive_qa`.
- `dataset`: The name of the Huggingface dataset used to load the model. For Qasper, this would be `allenai/qasper`. For datasets loaded from a local directory, skip this.
- `subset`: Some Huggingface datasets have a `subset` field; some do not. Leave this blank for datasets with no `subset`.
- `data_dir`: For datasets loaded from a local directory, this should be the name of the directory located under `data/processed`.
- `skip_splits`: Some datasets (for instance MSLR 2022) have a test split available but its outputs are blank. List any splits here that should be skipped during instantiation.
- `split_lookup`: The standard Huggingface dataset split names are `train`, `validation`, and `test`. Some datasets use different names for some folds. For instance, [SciRepEval](https://huggingface.co/datasets/allenai/scirepeval) uses `evaluation` instead of `test`. This field is a dict mapping from the canonical split names to the names for this particular dataset, if needed. for instance, for SciRepEval, we could use:

  ```yaml
  train: train
  validation: validation
  test: eval
  ```

- `standardize_splits`: By default, if some splits are missing, the instantiation code will re-balance the data to create 3 splits. In some cases this may be undesirable; if so, just set `standardize_splits: false` and the code will preserve whatever splits are available on Huggingface.
- `formatter`: Sometimes, there's no way to conveniently write a Jinja template to directly format a dataset as it appears in Huggingface. When this occurs, specify a formatter. The formatter's job is to take a `datasets.Dataset` as input, and return a list of instances as output, such that each output instance has the fields expected by the Jinja templates for this task. Formatters go in `tasks/formatters`. The `name` of the formatter should match a filename in `tasks/formatters`, with the `.py` extension removed. The formatter script should define a class `Formatter` with a method `format_instances`; this is the method that will take a `datasets.Dataset` and return a list of formatted instances. For an example, see the [qasper formatter](../sciriff/instructions/formatters/qasper.py).
  - The `args` to the formatter specify additional (optional) arguments that will be passed to the `Formatter` constructor. This can be useful if you want to use the same formatter for different tasks based on the same dataset.
- `metadata` for the task. See [task_schema.yaml](../sciriff/instructions/task_schema.yaml) for more information on metadata fields and their allowed values.

## Template details

Templates are numbered with ints starting from 0. Our current iteration of SciRIFF just uses the 0th template and ignores the rest; future work could explore using multiple templates. Each template supports the following two fields.

- `jinja`: The jinja template. Exactly the same as [promptsource](https://github.com/bigscience-workshop/promptsource); input and output are separated by `|||.` Since your Jinja template will likely be a multiline string, please use a [block scalar](https://yaml-multiline.info/) `|` to indicate a multiline string. For example:

  ```yaml
  ...
  <!-- Do this -->
  jinja: |
    For this task, please write a story about a miniature goldendoodle.

    The dog's name should be {{ name }}.

  <!-- Not this -->
  jinja: 'For this task, please write a story about a miniature goldendoodle.

    The dog's name should be {{ name }}.'
  ```

- `answer_choices`: Optionally, the list of answer choices, separated by `|||`. Exactly as in promptsource.
