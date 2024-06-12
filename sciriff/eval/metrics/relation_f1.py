from collections import Counter

from sciriff.eval.metrics import util


class RelationF1:
    "Exact-match relation F1."

    def __init__(self, tup_len, overlap_thresh):
        self.tup_len = tup_len  # Relation tuple length.
        self.overlap_thresh = overlap_thresh

    @staticmethod
    def _normalize(xs):
        "Normalize a single predicted list by converting to lowercase strings."
        try:
            res = [util.normalize_list_entry(x) for x in xs]
            return res
        except TypeError:
            return None

    def _parse_entry(self, entry):
        "Parse a single entry in the model's list of predictions."
        if not isinstance(entry, list):
            self.counts_parse["entry_not_list"] += 1
            return None
        elif len(entry) == self.tup_len:
            res = self._normalize(entry)
            if res is not None:
                self.counts_parse["good_list"] += 1
                return tuple(res)
            else:
                self.counts_parse["bad_list"] += 1
                return None
        else:
            self.counts_parse["wrong_length"] += 1
            return None

    def _parse_prediction(self, pred):
        """
        Parse a prediction, counting errors if they occur.
        """
        if pred == []:
            return []

        res = []

        try:
            is_list_of_lists = isinstance(pred[0], list)
        except TypeError:
            # This will trigger if the prediction is an int or other subscriptable
            # datatype.
            return []

        if is_list_of_lists:
            # If the first entry in the prediction is a list, attempt to parse each
            # entry. Throw out those that aren't lists, or that have the wrong length.
            for entry in pred:
                parsed = self._parse_entry(entry)
                if parsed is not None:
                    res.append(parsed)
        else:
            # Otherwise, maybe the model just returned a non-nested list; use it if
            # possible.
            parsed = self._parse_entry(pred)
            if parsed is not None:
                res.append(parsed)

        return res

    def _evaluate_f1_exact(self, pred, ref):
        "Count exact matches."
        pred_set = set(pred)
        ref_set = set(ref)
        self.counts_score["exact_match"] += len(pred_set & ref_set)

    def _is_substring_match(self, pred_entry, ref_entry):
        "Return True if every pred element is a sub / superstring of ref."
        if len(pred_entry) != self.tup_len or len(ref_entry) != self.tup_len:
            raise ValueError("Unexpected entry length.")
        for pred_item, ref_item in zip(pred_entry, ref_entry):
            if pred_item == "":
                return False
            if pred_item not in ref_item and ref_item not in pred_item:
                return False

        return True

    def _is_overlap(self, pred_entry, ref_entry):
        "Return True if every element has token F1 of at least `self.overlap_thresh`."
        if len(pred_entry) != self.tup_len or len(ref_entry) != self.tup_len:
            raise ValueError("Unexpected entry length.")
        for pred_item, ref_item in zip(pred_entry, ref_entry):
            if pred_item == "":
                return False
            if util.compute_token_f1(pred_item, ref_item) < self.overlap_thresh:
                return False

        return True

    def _evaluate_f1_fuzzy(self, sim_fn, sim_name, pred, ref):
        "Compute fuzzy F1 score given a similarity function `sim_fn`."
        already_used_refs = set()

        for pred_entry in pred:
            for ref_entry in ref:
                if ref_entry in already_used_refs:
                    continue
                if sim_fn(pred_entry, ref_entry):
                    self.counts_score[sim_name] += 1
                    already_used_refs.add(ref_entry)
                    # Continue on to the next prediction so we don't double-count.
                    break

    def _evaluate_one(self, instance):
        pred = self._parse_prediction(instance["pred"])
        ref = [tuple(self._normalize(entry)) for entry in instance["ref"]]

        self.counts_score["preds"] += len(pred)
        self.counts_score["refs"] += len(ref)
        self._evaluate_f1_exact(pred, ref)
        self._evaluate_f1_fuzzy(self._is_substring_match, "substring_match", pred, ref)
        self._evaluate_f1_fuzzy(self._is_overlap, "overlap_match", pred, ref)

    def evaluate(self, instances):
        self.counts_parse = util.count_dict(
            ["entry_not_list", "good_list", "bad_list", "wrong_length"]
        )
        self.counts_score = util.count_dict(
            ["preds", "refs", "exact_match", "substring_match", "overlap_match"]
        )

        for instance in instances:
            self._evaluate_one(instance)

        res = {}
        res["exact"] = util.compute_f1(
            self.counts_score["exact_match"],
            self.counts_score["preds"],
            self.counts_score["refs"],
        )
        res["substring"] = util.compute_f1(
            self.counts_score["substring_match"],
            self.counts_score["preds"],
            self.counts_score["refs"],
        )
        res["overlap"] = util.compute_f1(
            self.counts_score["overlap_match"],
            self.counts_score["preds"],
            self.counts_score["refs"],
        )
        self.counts_parse["frac_success"] = util.safe_div(
            self.counts_parse["good_list"], util.sum_dict(self.counts_parse)
        )
        res["parse_counts"] = self.counts_parse

        return res
