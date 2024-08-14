class Formatter:
    """
    Formatter for the MultiCite Dataset
    """

    def __init__(self):
        self.label_transformations = {
            "future_work": "FutureWork",
        }

    def transform_label(self, label):
        transformed_labels = []
        for lbl in label.split():
            # Use predefined label mapping or capitalize label by default
            transformed_label = self.label_transformations.get(lbl, lbl.capitalize())
            transformed_labels.append(transformed_label)
        return transformed_labels

    def format_instance(self, instance):
        context_with_citation = instance["x"]
        transformed_labels = self.transform_label(instance["y"])

        metadata = {"id": instance["id"]}

        # Return the transformed instance
        return {
            "_id": instance["_id"],
            "input_transformed": context_with_citation,
            "output_transformed": transformed_labels,
            "metadata": metadata,
        }

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
