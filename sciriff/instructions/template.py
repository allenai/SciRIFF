"""
Templating code. Adapted from https://github.com/bigscience-workshop/promptsource.
"""

import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined
from typing import Optional
from copy import deepcopy


# `StrictUndefined` will throw an error if a field is not defined in the template; this
# is better than silently leaving it blank.
env = Environment(loader=BaseLoader, undefined=StrictUndefined)
env.globals.update(zip=zip)


class Template(yaml.YAMLObject):
    """
    Class for a single prompt template.
    """

    text_var_length = 2048
    pipe_protector = "3ed2dface8203c4c9dfb1a5dc58e41e0"

    def __init__(
        self,
        jinja: str,
        answer_choices: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
        evaluate: bool = False,  # Is this the evaluation template for this task?
    ):
        self.jinja = jinja
        self.answer_choices = answer_choices
        self.name = name
        self.metadata = metadata
        self.evaluate = evaluate

    def get_answer_choices_expr(self):
        """
        Returns a Jinja expression for computing the answer choices from an example.

        :return: String, or None if no answer choices
        """
        return self.answer_choices

    def get_answer_choices_list(self, example):
        """
        Returns a list of answer choices for a given example

        :return: list of strings, or None if get_answer_choices_expr is None
        """
        jinja = self.get_answer_choices_expr()
        if jinja is None:
            return None

        rtemplate = env.from_string(jinja)
        protected_example = self._escape_pipe(example)
        rendered_choices = rtemplate.render(**protected_example)
        return [
            self._unescape_pipe(answer_choice.strip())
            for answer_choice in rendered_choices.split("|||")
        ]

    def apply(
        self, example_id, example, metadata, truncate=False, highlight_variables=False
    ):
        """
        Creates a prompt by applying this template to an example

        :param example_id: An ID for this example from the original data. Formatted like
            {split}_{index}.
        :param example: the dataset example to create a prompt for
        :param truncate: if True, example fields will be truncated to TEXT_VAR_LENGTH
            chars
        :param highlight_variables: highlight the added variables
        :param metadata: The metadata dictionary to include with each example
        :return: tuple of 2 strings, for prompt and output
        """
        jinja = self.jinja

        # Truncates the prompt if needed
        if truncate:
            # Escaping curly braces requires doubling them
            trunc_command = f" | string | truncate({self.text_var_length}) }}}}"
            jinja = jinja.replace("}}", trunc_command)

        # Highlights text that was substituted for variables, if requested
        if highlight_variables:
            jinja = jinja.replace("}}", " | highlight }}")
        rtemplate = env.from_string(jinja)

        protected_example = self._escape_pipe(example)
        metadata = deepcopy(metadata)
        metadata["template_metadata"] = {"use_for_eval": self.evaluate}
        # Adds in answer_choices variable
        if "answer_choices" in protected_example:
            raise ValueError("Example contains the restricted key 'answer_choices'.")

        protected_example["answer_choices"] = self.get_answer_choices_list(example)

        # Renders the Jinja template
        rendered_example = rtemplate.render(**protected_example)

        # Splits on the separator, and then replaces back any occurrences of the
        # separator in the original example
        parts = [
            self._unescape_pipe(part).strip() for part in rendered_example.split("|||")
        ]
        if len(parts) != 2:
            raise ValueError("Wrong number of parts after splitting.")

        return {
            "input": parts[0],
            "output": parts[1],
            "metadata": metadata,
            "_instance_id": example_id,
            "_template_id": self._id,
        }

    @classmethod
    def _escape_pipe(cls, example):
        # Replaces any occurrences of the "|||" separator in the example, which
        # which will be replaced back after splitting
        protected_example = {
            key: value.replace("|||", cls.pipe_protector)
            if isinstance(value, str)
            else value
            for key, value in example.items()
        }
        return protected_example

    @classmethod
    def _unescape_pipe(cls, string):
        # replaces back any occurrences of the separator in a string
        return string.replace(cls.pipe_protector, "|||")
