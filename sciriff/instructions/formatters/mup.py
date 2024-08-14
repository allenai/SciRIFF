import spacy

class Formatter:
    """
    Counts the the number of sentences in the answer and includes this as part of the
    prompt. Otherwise the task is underspecified.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def format_instance(self, instance):
        n_sents = len([sent for sent in self.nlp(instance["summary"]).sents])
        instance["summary_sentences"] = n_sents

        return instance

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
