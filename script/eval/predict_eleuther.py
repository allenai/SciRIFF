import beaker
from beaker import Beaker, ExperimentSpec, TaskSpec
from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess

from sciriff.lib import paths


def make_parser():
    desc = "Kicks off Beaker jobs to make predictions for all science-eval tasks."
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "--model",
        type=str,
        help="Model. Can be vllm, openai, etc.",
        choices=["vllm", "openai-chat-completions", "anthropic"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="""This argument depends on what you gave for the `model` argument.
        If `model` is `vllm`, this can be either (1) a path to a directory with model
        weights, or (2) the name of a Huggingface model.
        If `model` is openai or anthropic, this should be the name of the model you
        want to use (e.g. `gpt-4-turbo`)""",
    )
    parser.add_argument(
        "--model_display_name",
        type=str,
        help="""If given, use this name as the output directory for predictions.
        Otherwise, pick a reasonable name based on the `model_name` argument.""",
        default=None,
    )
    parser.add_argument(
        "--chat_template",
        default="general",
        type=str,
        help="Apply the chat template in the directory `eleuther_templates/[chat_template]`.",
    )
    parser.add_argument(
        "--tasks",
        help="""Tasks to make predictions for. If not given, do all. If `science_adapt`
        is given, predict all tasks in one process.""",
        type=str,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="Number of GPUs to use. Use 1 for 7B and 4 for 70B",
        default=0,
    )
    parser.add_argument(
        "--result_base",
        type=str,
        help="""Base directory for results; this script will create a subdirectory for
        the specific model being evaled""",
        default=str(paths.project_root / "results" / "by_model"),
    )
    parser.add_argument(
        "--limit", type=int, help="Max instances per task.", default=None
    )
    ####################
    # All arguments below this point only apply to Beaker batch jobs; ignore if not
    # AI2-internal.
    parser.add_argument(
        "--beaker",
        action="store_true",
        help="""If given, run this job using beaker; for internal use.""",
    )
    parser.add_argument(
        "--beaker_cluster",
        type=str,
        default=None,
        help="Cluster to use. If multiple, separate with commas.",
    )
    parser.add_argument(
        "--beaker_dataset",
        help=(
            "If given, specifies the ID of the Beaker dataset where the model is stored."
        ),
        default=None,
    )
    parser.add_argument(
        "--beaker_budget",
        type=str,
        help="AI2 budget name that this experiment goes toward.",
    )
    parser.add_argument(
        "--beaker_workspace",
        help="Beaker workspace for this experiment.",
        type=str,
    )
    parser.add_argument(
        "--beaker_image",
        help="Beaker image to use for evaluation",
        type=str,
    )
    parser.add_argument(
        "--beaker_priority",
        help="Priority for Beaker jobs.",
        type=str,
        default="normal",
    )

    return parser


def get_eval_tasks(args):
    "Get evaluation tasks."
    eval_task_dir = paths.EVAL_DIR / "eleuther_templates/tulu"
    names = [p.stem for p in eval_task_dir.glob("*.yaml")]
    names = [name for name in names if name != "_default_template"]

    # If requested, run only user-specified tasks.
    if args.tasks is not None:
        if args.tasks == ["science_adapt"]:
            return args.tasks
        for task in args.tasks:
            # Make sure all the tasks are available.
            if task not in names:
                raise ValueError(f"Unrecognized task {task} requested.")
        return args.tasks
    else:
        return names


def get_beaker_settings(args):
    if args.beaker_cluster is None:
        cluster = [
            "ai2/s2-cirrascale",
            "ai2/s2-cirrascale-l40",
            "ai2/allennlp-cirrascale",
        ]
    else:
        cluster = args.beaker_cluster.split(",")

    env_vars = [
        # Get Huggingface home dir from environment.
        beaker.EnvVar(name="HF_HOME", value=os.environ.get("HF_HOME")),
        # OpenAI and Anthropic keys
        beaker.EnvVar(name="OPENAI_API_KEY", secret="openai_api_key"),
        beaker.EnvVar(name="OPENAI_API_SECRET_KEY", secret="openai_api_key"),
        beaker.EnvVar(name="ANTHROPIC_API_KEY", secret="anthropic_api_key"),
    ]

    # Set Beaker datasets.
    datasets = [  # Always mount NFS.
        beaker.DataMount(
            source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
            mount_path="/net/nfs.cirrascale",
        )
    ]
    if args.beaker_dataset is not None:
        this_dataset = beaker.DataMount(
            source=beaker.DataSource(beaker=args.beaker_dataset),
            mount_path=args.model_name,
        )
        datasets.append(this_dataset)

    return {
        "image": beaker.ImageSource(beaker=args.beaker_image),
        "result": beaker.ResultSpec(path="/unused"),
        "datasets": datasets,
        "context": beaker.TaskContext(priority=beaker.Priority(args.beaker_priority)),
        "constraints": beaker.Constraints(cluster=cluster),
        "env_vars": env_vars,
        "resources": beaker.TaskResources(gpu_count=args.gpus),
    }


def make_task_command(task_name, result_dir, args):
    "Create Beaker task command to run evaluation on each dataset."
    result_subdir = result_dir / task_name
    result_subdir.mkdir(exist_ok=True, parents=True)
    if any(result_subdir.iterdir()):
        # If there are already files here, skip it.
        return None

    result_file = result_subdir / "eleuther.jsonl"

    # Get the prompt directory.
    include_path = paths.EVAL_DIR / f"eleuther_templates/{args.chat_template}"

    # Construct model args.
    if args.model == "vllm":
        model_args = f"pretrained={args.model_name},tensor_parallel_size={args.gpus},dtype=float16"
    else:
        model_args = f"model={args.model_name}"

    command = [
        "python",
        "-m",
        "lm_eval",
        "--include_path",
        include_path,
        "--model",
        args.model,
        "--model_args",
        model_args,
        "--gen_kwargs",
        "max_gen_toks=1024",  # TODO(dwadden) Should make this modifiable.
        "--tasks",
        task_name,
        "--batch_size",
        "1",
        "--predict_only",
        "--log_samples",
        "--output_path",
        result_file,
    ]
    if args.limit is not None:
        command += ["--limit", args.limit]

    return [str(x) for x in command]


def kickoff(args):
    if args.model_display_name is not None:
        model_stem = args.model_display_name
    else:
        if args.model == "vllm":
            model_stem = Path(args.model_name).name
        else:
            model_stem = args.model_name

    result_dir = Path(args.result_base) / model_stem
    result_dir.mkdir(exist_ok=True, parents=True)

    # Loop over datasets and create a Beaker task for each.
    beaker_settings = get_beaker_settings(args)

    experiment_tasks = []
    for task_name in get_eval_tasks(args):
        task_command = make_task_command(task_name, result_dir, args)
        if task_command is None:
            continue
        this_task = TaskSpec(
            name=f"eval-{model_stem}-{task_name}",
            command=task_command,
            **beaker_settings,
        )
        experiment_tasks.append(this_task)

    # If working interactively, run the evals one at a time in the current session.
    if not args.beaker:
        for task_spec in experiment_tasks:
            print(f"Running {task_spec.name}.")
            subprocess.run(task_spec.command)

    # Otherwise kick off batch jobs.
    else:
        spec = ExperimentSpec(
            description=f"Science adapt Eleuther prediction for {model_stem}.",
            tasks=experiment_tasks,
            budget=args.beaker_budget,
        )

        # Make the experiment and run it.
        if args.beaker_workspace is not None:
            beaker_client = Beaker.from_env(default_workspace=args.beaker_workspace)
        else:
            beaker_client = Beaker.from_env()

        beaker_client.experiment.create(spec=spec)
        print(f"Kicked off evaluation for {model_stem}.")


def main():
    parser = make_parser()
    args = parser.parse_args()
    kickoff(args)


if __name__ == "__main__":
    main()
