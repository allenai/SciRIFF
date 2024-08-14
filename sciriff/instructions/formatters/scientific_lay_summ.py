class Formatter:
    """
    Formatter for Scientific Lay Summarization Dataset
    """

    def __init__(self):
        self.punctuation = {
            ".",
            ",",
            ";",
            ":",
            "!",
            "?",
            ")",
            "]",
            "}",
            "-",
            "—",
            "…",
            "(",
            "[",
            "{",
        }

    def clean_spaces_before_punctuation(self, text):
        cleaned_text = ""
        for i, char in enumerate(text):
            if char in self.punctuation and i > 0 and text[i - 1] == " ":
                cleaned_text = cleaned_text[:-1]
            cleaned_text += char
        return cleaned_text

    def format_instance(self, instance):
        # Just use the abstract, intro, and discussion. This is often short
        # enough to be useable. The supplemental material at end is really long.

        sections = instance["article"].split("\n")
        section_headings = instance["section_headings"].split("\n")

        if len(section_headings) < 4:
            return None
        if (
            section_headings[0] != "Abstract"
            or section_headings[1] != "Introduction"
            or section_headings[3] != "Discussion"
        ):
            return None
        if len(sections) != len(section_headings):
            return None

        sections = sections[:2] + sections[3:4]
        section_headings = section_headings[:2] + section_headings[3:4]

        combined_sections = [
            f"{heading}:\n{text}" for heading, text in zip(section_headings, sections)
        ]
        input_transformed = "\n".join(combined_sections)

        input_transformed = self.clean_spaces_before_punctuation(input_transformed)

        summary = instance["summary"]
        metadata = {
            "title": instance["title"],
            "year": instance["year"],
            "keywords": instance["keywords"].replace("\n", ", "),
        }
        return {
            "_id": instance["_id"],
            "input_transformed": input_transformed,
            "summary": summary,
            "metadata": metadata,
        }

    def format_instances(self, dataset):
        res = []
        for instance in dataset:
            formatted = self.format_instance(instance)
            if formatted is not None:
                res.append(formatted)

        return res
