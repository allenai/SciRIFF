class Formatter:
    """
    Some of the abstracts for `scientific_papers` are blank; skip them.
    """
    def __init__(self):
        self.min_toks = 50  # Discard instances with abstract or article below 50.

    def format_instance(self, instance):
        for field in ["article", "abstract"]:
            n_toks = len(instance[field].split())
            if n_toks < self.min_toks:
                return None
            else:
                return instance

    def format_instances(self, dataset):
        res = []
        for instance in dataset:
            formatted = self.format_instance(instance)
            if formatted is not None:
                res.append(formatted)

        return res
