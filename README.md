# SciRIFF

This repository contains code for the preprint [SciRIFF: A Resource to Enhance Language Model Instruction-Following over Scientific Literature](https://arxiv.org/abs/2406.07835). SciRIFF is a collection of 54 tasks targeting instruction-following over scientific literature. Tasks were created by converting existing scientific datasets to a common instruction-following format via expert-written templates.

The SciRIFF dataset, as well as the SciTulu models trained on SciRIFF, are available in the Hugging Face [SciRIFF collection](https://huggingface.co/collections/allenai/sciriff-665f61ba7315e1d202e5f6bf). This repository contains code to evaluate the SciTulu models on 9 held-out SciRIFF tasks, as well as details explaining how to use the data to train new models. Shortly, we will add templates for all tasks, as well as code to recreate the dataset using these templates.

**Table of Contents**

- [Setup](#setup)
- [Evaluation](#evaluation)
- [Model training](#model-training)
- [Recreating the dataset](#recreating-the-dataset)
- [The SciRIFF collection](#the-sciriff-collection)

## Setup

We recommend using a Conda env:

```bash
conda create --name sciriff python=3.11
conda activate sciriff
```

We use the Eleuther harness to handle inference for evaluation. For stability, it's best to install a specific commit, as follows:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout e74ec96
pip install -e .[vllm]
```

Then, install `sciriff` by `cd`ing back to the project directory and running:

```bash
pip install -e .
```

You may get a warning about incompatible versions of the `typer` package; this can safely be ignored.

For two of our evaluations, we use GPT-3.5 as an LM judge. In order to do these evaluations, you'll need an OpenAI API key:

```bash
export OPENAI_API_KEY=[your_openai_key]
```

## Evaluation

To evaluate, we first use the Eleuther harness to handle inference, and then run evluations on the results. For implementation details, see [evaluation.md](doc/evaluation.md). For examples of each evaluation task, see [evaluation_tasks.md](doc/evaluation_tasks.md)

### Making predictions

Use `predict_eleuther.py` to make predictions for all eval tasks. The example below makes predictions using SciTulu-7B. The results will go in `results/predictions/scitulu-7b`.

For the 7B, you should be fine using a single A6000 gpu. For the 70B, we've generally used 4 80GB A100's or similar, but it may be possible to do with less. Inference on the whole eval set will take a few hours; you can use the `--limit` flag to cap the number of instances per task.

```bash
python script/eval/predict_eleuther.py \
  --model vllm \
  --model_name allenai/scitulu-7b \
  --chat_template tulu \
  --gpus 1 \
  --tasks science_adapt \
  --result_base results/predictions
```

To make predictions with an API model, you can do:

```bash
python script/eval/predict_eleuther.py \
    --model=openai-chat-completions \
    --model_name=gpt-3.5-turbo-1106 \
    --chat_template general \
    --tasks science_adapt \
    --result_base results/predictions \
    --limit 50
```

### Computing metrics

Run `compute_science_metrics.py` to compute metrics based on the model predictions.

```bash
python script/eval/compute_science_metrics.py \
  --pred_dir results/predictions \
  --metrics_dir results/metrics
```

If you've run predictions `predict_eleuther.py` on multiple models, this will evaluate all models for which predictions are available under `results/predictions`. Metrics for each model will be saved to `results/metrics/by_model/{model_name}`; there will be a subfolder for each task including cleaned-up predictions and detailed metrics.

Metrics from all models will be collected in `results/metrics/tables`. The file `results/metrics/tables/summary.tsv` provides a summary of the metrics for all tasks.

The results in `summary.tsv` for SciTulu should match the results reported in the preprint within roughly a point; there may be small differences due to slightly different software versions or (for LM judge evals) changes in OpenAI API models.

## Model training

If you'd like to train a new model using the SciTulu data, you can use the [Open Instruct](https://github.com/allenai/open-instruct) repo. The training mix described in the preprint -- which uses 1000 instances / SciRIFF task combined with a matching about of data from the Tulu-V2 mix -- is available on Huggingface as [https://huggingface.co/datasets/allenai/SciRIFF-train-mix/tree/main](SciRIFF-train-mix), and is formatted to support Tulu training.

Once you've pulled and installed the open-instruct code, you can kick off a training run as follows:

```bash
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 8 \
  --use_deepspeed \
  --deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --model_name_or_path llama_2_7b \
  --use_flash_attn \
  --tokenizer_name llama_2_7b \
  --dataset_name allenai/SciRIFF-train-mix \
  --use_slow_tokenizer \
  --max_seq_length 4096 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --num_train_epochs 2 \
  --output_dir output/ \
  --with_tracking \
  --report_to tensorboard \
  --logging_steps 1
```

You can also use the open-instruct code to run evals on general instruction-following tasks (e.g. MMLU, GSM8K, etc); see the repo for more details.

## Recreating the dataset

We will provide templates and scripts shortly to recreate the dataset.

## The SciRIFF collection

The [SciRIFF](https://huggingface.co/collections/allenai/sciriff-665f61ba7315e1d202e5f6bf) collection on Hugging Face consists of the following:

- [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF): The full SciRIFF dataset as described in our [preprint](https://arxiv.org/abs/2406.07835).
- [SciRIFF-train-mix](https://huggingface.co/datasets/allenai/SciRIFF-train-mix) The training mix used to train our SciTulu models; also described in the [preprint](https://arxiv.org/abs/2406.07835).
- [SciTulu 7B](https://huggingface.co/allenai/scitulu-7b): Checkpoint for the SciTulu 7B model.
- [SciTulu 70B](https://huggingface.co/allenai/scitulu-70b): Checkpoint for the SciTulu 70B model.
