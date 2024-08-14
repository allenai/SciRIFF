import json
import re


class Formatter:
    """
    Format ChemSum. This is a simple formatter to construct output data structures.
    """

    def __init__(self):
        self.split_tok = "<!>"

    def format_instance(self, instance):
        headers = instance["headers"].split(self.split_tok)
        sections = instance["sections"].split(self.split_tok)

        if len(headers) != len(sections):
            raise ValueError("The number of headers and sections are not equal.")

        article_text = ""
        for header, section in zip(headers, sections):
            article_text += header + "\n" + section + "\n\n"

        # Strip final newline.
        article_text = article_text[:-1]

        instance["article_text"] = article_text
        instance["abstract"] = instance["abstract"].strip()

        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
