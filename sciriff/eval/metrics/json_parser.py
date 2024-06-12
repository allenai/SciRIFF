"""
Extracts json from model answers, keeping track of how many failed.
"""

import json
from sciriff.eval.metrics import util
from collections import Counter


class JSONParser:
    "Extract json from response. If no json is found, return an empty response."

    def __init__(self, default, task) -> None:
        self.default = default
        self.task = task

    def _find_json(self, text):
        """
        Find and parse the first valid JSON appearing in a mixed text.
        """
        if isinstance(self.default, dict):
            start_delim, end_delim = ("{", "}")
        elif isinstance(self.default, list):
            start_delim, end_delim = ("[", "]")
        else:
            raise ValueError("Unexpected default type.")

        depth = 0
        start_index = -1

        for i, char in enumerate(text):
            if char == start_delim:
                if depth == 0:
                    start_index = i
                depth += 1
            elif char == end_delim:
                depth -= 1
                if depth == 0 and start_index != -1:
                    try:
                        json_obj = json.loads(text[start_index : i + 1])
                    except json.JSONDecodeError:
                        start_index = -1  # Reset the start index and keep going.
                    else:
                        return json_obj, "extract_success"

        # If we get to the end without finding anything, return default.
        return self.default, "extract_failure"

    def apply_instance(self, inst):
        # Fix strange edge case for tulu models on evidence inference: they sometimes
        # close with a } instead of ].
        if self.task == "EvidenceInference":
            inst = inst.strip()
            if len(inst) and inst[-1] == "}":
                inst = inst[:-1] + "]"

        try:
            filtered_resp = json.loads(inst)
            if filtered_resp is None or filtered_resp == "null":
                # The second case above happens when the response is `'"null"'`.
                filtered_resp = self.default
                status = "null_json"
                self.counts[status] += 1
            else:
                status = "valid_json"
                self.counts[status] += 1
            return {"raw": inst, "value": filtered_resp, "status": status}
        except json.JSONDecodeError:
            filtered_resp, status = self._find_json(inst)
            self.counts[status] += 1
            return {"raw": inst, "value": filtered_resp, "status": status}

    def __call__(self, instances):
        # Keep track of the final status for each. We don't use this right now, but
        # could at some point.
        self.counts = util.count_dict(
            ["valid_json", "null_json", "extract_success", "extract_failure"]
        )

        res = []
        for inst in instances:
            formatted_inst = self.apply_instance(inst)
            res.append(formatted_inst)

        self.counts["frac_success"] = util.safe_div(
            self.counts["valid_json"] + self.counts["extract_success"],
            util.sum_dict(self.counts),
        )

        return res, self.counts
