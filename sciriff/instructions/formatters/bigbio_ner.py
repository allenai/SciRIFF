import json


class Formatter:
    """
    Formatter for Named Entity Recognition from BigBIO datasets
    """

    def __init__(self, types_map):
        self.types_map = dict([x.split(":") for x in types_map.split(",")])
        self.ner_types = list(set(self.types_map.values()))

    # TODO: Add more deduplication heuristics for entities
    # TODO: Substring issue in PICO gold entities needs to be fixed
    def format_instance(self, instance):
        # Under HF normalized schema for BigBIO, abstract text is stored as a list of passages
        # Sometimes each passage is a list of sentences
        # TODO: Run for all eval datasets to verify this works
        abstract = [" ".join(x["text"]).strip() for x in instance["passages"]]
        abstract = " ".join(abstract)

        entity_dict = {x: [] for x in self.ner_types}
        for entity in instance["entities"]:
            entity_dict[self.types_map[entity["type"]]].append(" ".join(entity["text"]))

        # Remove repeated occurrences of same entity string
        entity_dict = {k: list(set(v)) for k, v in entity_dict.items()}

        formatted_instance = {
            "_id": instance["_id"],
            "ner_type_list": self.ner_types,
            "abstract": abstract,
            "ner_dict": entity_dict,
        }

        return formatted_instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
