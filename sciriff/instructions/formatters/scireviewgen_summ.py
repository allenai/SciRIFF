class Formatter:
    """
    Formatter for SciReviewGen Dataset
    """

    def format_instance(self, instance):
        parts = instance["reference"].split("<s>", 2)
        literature_review_title = parts[0]
        chapter_title = parts[1]

        abs_parts = parts[2].split("<s> BIB")

        # Need to handle the final element separately, noqa: E501
        abs_bib_tup = [part.split(" </s>", 1) for part in abs_parts[:-1]]

        # Get abstracts from the second part of each element
        abstracts = [
            pair[1].rsplit("<s>", 2)[-1] for pair in abs_bib_tup if len(pair) > 1
        ]
        bib_labels = [f"BIB{pair[0]}" for pair in abs_bib_tup if len(pair) > 1]

        combined_context = f"Literature Review Title: {literature_review_title}\n"
        combined_context += f"Chapter Title: {chapter_title}\n"
        for i, (abstract, bib_label) in enumerate(zip(abstracts, bib_labels)):
            combined_context += (
                f"{i+1}. Abstract of Cited Paper ({bib_label}): {abstract}\n"
            )

        # Transformed instances
        formatted_instance = {
            "_id": instance["_id"],
            "input_transformed": combined_context.strip(),
            "target_transformed": instance["target"],
        }
        return formatted_instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
