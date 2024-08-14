import json

class Formatter:
    """
    Format SciERC.
    """

    def __init__(self):
        self.entity_types = [
            "Task",
            "Method",
            "Metric",
            "Material",
            "OtherScientificTerm",
            "Generic",
        ]
        self.relation_types = [
            "USED-FOR",
            "EVALUATE-FOR",
            "FEATURE-OF",
            "HYPONYM-OF",
            "CONJUNCTION",
            "COMPARE",
            "PART-OF",
        ]

    @staticmethod
    def _fix_whitespace(instance):
        """
        There are a few formatting errors where the span isn't found in the input; try
        to fix, or return None if we can't.
        """
        def attempt_fix(mention):
            attempted_replacements = [(" - ", "-"), (" , ", ", "), (" 's", "'s")]
            for replacement in attempted_replacements:
                if replacement[0] in mention:
                    return mention.replace(replacement[0], replacement[1])

            return mention

        the_input = instance["org_text"]
        ner = instance["ner_answer"]
        rel = instance["re_answer"]

        for this_ner in ner:
            if this_ner[0] not in the_input:
                old_ner = this_ner[0]
                new_ner = attempt_fix(old_ner)
                if new_ner in the_input:
                    this_ner[0] = new_ner
                else:
                    return None

        for this_rel in rel:
            for i in range(2):
                if this_rel[i] not in the_input:
                    old_rel = this_rel[i]
                    new_rel = attempt_fix(old_rel)
                    if new_rel in the_input:
                        this_rel[i] = new_rel
                    else:
                        return None

        return instance

    def format_instance(self, instance):
        # Calling dict() on list of pairs converts to a dict.
        instance["ner_pipe_str"] = " || ".join(
            [f"{x[0]}: {x[1]}" for x in instance["ner_answer"]]
        )
        entity_dict = {k: [] for k in self.entity_types}
        for ent_name, ent_type in instance["ner_answer"]:
            entity_dict[ent_type].append(ent_name)
        instance["ner_dict"] = entity_dict

        # Reorder so the relation goes last.
        re_answer = []
        for entry in instance["re_answer"]:
            to_append = [entry[0], entry[2], entry[1]]
            re_answer.append(to_append)

        instance["re_answer"] = re_answer

        # Fix cases of whitespace errors. If un-fixable, return None.
        instance = self._fix_whitespace(instance)
        return instance

    def format_instances(self, dataset):
        res = []
        for instance in dataset:
            formatted = self.format_instance(instance)
            if formatted is not None:
                res.append(formatted)

        return res
