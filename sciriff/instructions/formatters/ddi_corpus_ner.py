class Formatter:
    def extract_distinct_ner_types(self, dataset):
        ner_types = set()
        for instance in dataset:
            for entity in instance["entities"]:
                ner_types.add(entity["type"])
        return sorted(list(ner_types))

    def format_instance(self, instance, distinct_ner_types):
        ner_answer = [
            f"{entity['text']}: {entity['type']}" for entity in instance["entities"]
        ]
        ner_answer = list(set(ner_answer))

        formatted_instance = {
            "document_id": instance["document_id"],
            "org_text": instance["text"],
            "ner_answer": ner_answer,
            "distinct_ner_types": distinct_ner_types,
        }
        return formatted_instance

    def format_instances(self, dataset):
        distinct_ner_types = self.extract_distinct_ner_types(dataset)
        formatted_instances = []
        for instance in dataset:
            formatted_instances.append(
                self.format_instance(instance, distinct_ner_types)
            )
        return formatted_instances
