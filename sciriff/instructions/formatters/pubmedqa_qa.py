class Formatter:
    """
    Formatter for PubMedQA Dataset
    """

    def format_instance(self, instance):
        contexts = instance["CONTEXTS"]
        context_labels = instance["LABELS"]
        formatted_contexts = []
        for label, context in zip(context_labels, contexts):
            formatted_contexts.append(f"({label}) {context}")

        # Combine formatted contexts into a single string
        combined_context = "\n".join(formatted_contexts)

        # Extract question and answer
        question = instance["QUESTION"]
        answer = instance["final_decision"].upper()

        formatted_instance = {
            "_id": instance["_id"],
            "context": combined_context,
            "question": question,
            "answer": answer,
        }
        return formatted_instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
