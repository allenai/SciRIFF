import warnings


class Formatter:
    """
    Formatter for Named Entity Recognition from GENIA dataset.
    This uses a separate formatter because we're using the
    preprocessed version of GENIA from DyGIE++ instead of BigBIO.
    """

    def __init__(self, types_map):
        self.types_map = dict([x.split(":") for x in types_map.split(",")])
        self.ner_types = list(set(self.types_map.values()))
        self.failed_counter = 0

    def format_instance(self, instance):
        abstract = instance["org_text"]

        entity_dict = {x: [] for x in self.ner_types}
        for entity in instance["ner_answer"]:
            ename = ("").join(entity.split(":")[0:-1]).strip()
            etype = entity.split(":")[-1].strip()
            try:
                entity_dict[self.types_map[etype]].append(ename)
            except KeyError:
                # Ignore entities with no type annotation "name: type"
                # E.g. example_41 in dev set "breast cancer: MCF - 7) and negative (ER - "
                self.failed_counter += 1
        entity_dict = {k: list(set(v)) for k, v in entity_dict.items()}

        formatted_instance = {
            "_id": instance["_id"],
            "ner_type_list": self.ner_types,
            "abstract": abstract,
            "ner_dict": entity_dict,
        }

        return formatted_instance

    def format_instances(self, dataset):
        res = [self.format_instance(instance) for instance in dataset]
        if self.failed_counter:
            msg = f"Total of {self.failed_counter} entities skipped for GENIA_NER due to missing type annotation."
            warnings.warn(msg)
        return res
