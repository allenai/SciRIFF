class Formatter:
    """
    Formatter for ACL-ARC Dataset for Citation Intent Classification
    """

    def __init__(self):
        label_mapping = {
            "Future": "FutureWork",
            "Background": "Background",
            "Extends": "Extends",
            "Uses": "Uses",
            "Motivation": "Motivation",
            "CompareOrContrast": "CompareOrContrast",
        }

        self.label_mapping = label_mapping

    def reconstruct_sentence(self, sentence_list):
        res = []
        for word in sentence_list:
            original_text = word["originalText"]
            # Handle punctuations
            if word["after"].strip() and not word["after"].strip() in {
                ".",
                ",",
                ";",
                ":",
                "!",
                "?",
            }:
                space_after = " " + word["after"]
            else:
                space_after = word["after"]
            res.append(original_text + space_after)
        return "".join(res).strip()

    def format_instance(self, instance, all_distinct_labels):
        citation_sentence = instance.get("text", "")
        section = instance.get("section_title", "") or instance.get("section_name", "")

        context_before = []
        for sent in instance.get("sents_before", []):
            context_before.append(self.reconstruct_sentence(sent))
        context_before = " ".join(context_before).strip()

        context_after = []
        for sent in instance.get("sents_after", []):
            context_after.append(self.reconstruct_sentence(sent))
        context_after = " ".join(context_after).strip()

        label = instance.get("intent")
        label = self.label_mapping[label]

        return {
            "_id": instance["_id"],
            "citation_sentence": citation_sentence,
            "section": section,
            "context_before": context_before,
            "context_after": context_after,
            "label": label,
            "all_distinct_labels": all_distinct_labels,
        }

    def format_instances(self, dataset):
        all_distinct_labels = list(self.label_mapping.values())
        return [
            self.format_instance(instance, all_distinct_labels) for instance in dataset
        ]
