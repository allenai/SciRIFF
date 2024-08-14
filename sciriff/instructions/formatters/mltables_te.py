import json

TEMPLATE_ATTRIBUTES = {
    "Result": ["value", "type", "task", "metric", "training data/set", "test data/set", "model", "model settings", "experimental settings"],
    "Data Stat.": ["value", "type", "dataset", "attribute name", "sub-set/group name", "dataset features"],
    "Hyper-parameter/Architecture": ["value", "type", "model", "parameter/architecture name", "dataset"],
    "Other": ["value", "type"]
}

class Formatter:
    """
    Formatter for MLTables Dataset in the Schema-to-JSON benchmark
    """
    def format_instance(self, instance):
        table_code = instance["table_code"]
        sup_text = instance["sup_text"]

        table_code_text = f"{sup_text}\n\n{table_code}" if sup_text.strip() else table_code

        cell_record_str_list = []
        for each_record in instance["gold_json_records"]["cell_record"]:
            cell_record = json.loads(each_record)

            # Process properties, whose values are lists
            cell_record_processed = {}
            for key in TEMPLATE_ATTRIBUTES[cell_record["type"]]:
                assert key in cell_record, f"attribute {key} not found in cell record {cell_record}"
                value = cell_record[key]
                if isinstance(value, list):
                    # Select the value that has the most word overlap with the table code
                    value_select = sorted([(value_item, sum([word in table_code.lower() for word in value_item.lower().split()])/len(value_item.lower().split())) for value_item in value], key=lambda x: x[1], reverse=True)[0][0]
                    cell_record_processed[key] = value_select
                else:
                    cell_record_processed[key] = value
            cell_record_str = json.dumps(cell_record_processed, ensure_ascii=False)
            cell_record_str_list.append(cell_record_str)

        json_records = "\n".join(cell_record_str_list)

        formatted_instance = {
            "_id": instance["_id"],
            "table_code_text": table_code_text,
            "json_records": json_records
        }
        return formatted_instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
