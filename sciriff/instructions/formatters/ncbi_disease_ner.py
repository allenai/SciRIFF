import json


class Formatter:
    """
    Format the given dataset. This formatter constructs output data structures
    and makes the data human-readable by marking the entities in the text.
    """

    def format_instance(self, instance):
        # Combine tokens based on the NER tags
        tokens_with_tags, entities = self.get_tokens_with_tags(
            instance["tokens"], instance.get("ner_tags", [])
        )

        # Store the formatted sentence with the tags embedded
        instance["sentence_with_tags"] = " ".join(tokens_with_tags)

        # Store the entities with their tags as a string or return '[]' if no entities are found
        instance["only_tags"] = (
            ", ".join([f"{entity['entity']} <{entity['tag']}>" for entity in entities])
            if entities
            else "[]"
        )

        # For future usage, convert this instance into JSON format
        instance["formatted_json"] = json.dumps(
            {"id": instance["id"], "all_tags": instance["only_tags"]}
        )

        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]

    def get_tokens_with_tags(self, tokens, ner_tags):
        i = 0
        combined_tokens = []
        entities = []

        # Updated tag mapping for the new dataset
        tag_mapping = {0: "O", 1: "B-Disease", 2: "I-Disease"}

        while i < len(tokens):
            current_tag = tag_mapping.get(ner_tags[i], "O")
            if "B-" in current_tag:
                entity = [tokens[i]]
                i += 1
                while i < len(tokens) and "I-" in tag_mapping.get(ner_tags[i], ""):
                    entity.append(tokens[i])
                    i += 1
                entity_text = " ".join(entity)
                combined_tokens.append(entity_text)
                entities.append(
                    {"entity": entity_text, "tag": current_tag.split("-")[1]}
                )
            else:
                combined_tokens.append(tokens[i])
                i += 1

        return combined_tokens, entities
