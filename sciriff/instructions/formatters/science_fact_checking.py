import json


class Formatter:
    """
    Format science fact-checking datasets including SciFact, HealthVer, and CovidFact.
    This is a simple formatter to construct output data structures.
    """

    def format_instance(self, instance):
        sents_with_ids = [
            f"[{i}] {sent}" for i, sent in enumerate(instance["abstract"])
        ]
        instance["abstract_with_ids"] = " ".join(sents_with_ids)
        instance["abstract_with_newlines"] = "\n".join(instance["abstract"])
        instance["output_json_with_ids"] = json.dumps(
            {"verdict": instance["verdict"], "evidence": instance["evidence"]}
        )
        evidence_sents = [
            sent
            for i, sent in enumerate(instance["abstract"])
            if i in instance["evidence"]
        ]
        instance["output_json_with_sentences"] = json.dumps(
            {"verdict": instance["verdict"], "evidence": evidence_sents}
        )
        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
