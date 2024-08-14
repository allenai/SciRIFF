from sciriff.lib import util


class Formatter:
    """
    Format the drug combination extraction dataset.
    """

    @staticmethod
    def format_relation(rel, spans):
        res = []

        # Get all the drug names.
        for span_id in rel["spans"]:
            span_text = spans[span_id]["text"]
            res.append(span_text)

        # Get the relation class.
        res.append(rel["class"])

        return res

    def format_instance(self, instance):
        spans = util.list_to_dict(instance["spans"], key="span_id")
        relations = []

        for rel in instance["rels"]:
            relations.append(self.format_relation(rel, spans))

        instance["relations"] = relations

        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
