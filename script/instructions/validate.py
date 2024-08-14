"""
Validate templates according to task schema.
"""

import yaml
from sciriff.lib import paths
import jsonschema

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Validate templates according to task schema."
    )
    parser.add_argument(
        "--contributor",
        default=None,
        type=str,
        help="If given, only check tasks for this contributor.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    contributor = args.contributor
    schema_file = yaml.full_load(paths.TASK_SCHEMA_FILE.open())

    n_templates = len(list(paths.TEMPLATE_DIR.iterdir()))
    for template_file in paths.TEMPLATE_DIR.iterdir():
        template = yaml.full_load(template_file.open())
        if contributor:
            if contributor not in template["metadata"].get("contributor", ""):
                continue

        template_name = template_file.stem
        try:
            jsonschema.validate(template["metadata"], schema_file)
        except jsonschema.ValidationError as e:
            print(f"Failed on {template_name}")
            raise e

    print(f"Successfully validated {n_templates} templates.")


if __name__ == "__main__":
    main()
