class Formatter:
    """
    Formatter for Multi-XScience Dataset
    """

    def format_instance(self, instance):
        main_abstract = instance["abstract"]

        cited_abstracts_transformed = ""
        for i, abstract in enumerate(instance["ref_abstract"]["abstract"], start=1):
            cited_abstracts_transformed += f"[{i}]: {abstract}\n\n"

        related_work_transformed = instance["related_work"]
        for i, cite_marker in enumerate(instance["ref_abstract"]["cite_N"], start=1):
            related_work_transformed = related_work_transformed.replace(
                cite_marker, f"[{i}]"
            )

        return {
            "_id": instance["_id"],
            "main_abstract": main_abstract,
            "cited_abstracts_transformed": cited_abstracts_transformed.strip(),
            "related_work_transformed": related_work_transformed,
        }

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
