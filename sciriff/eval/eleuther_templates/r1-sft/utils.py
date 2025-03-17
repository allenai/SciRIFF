from functools import partial


def filter_task(ds, task):
    return ds.filter(lambda inst: inst["_instance_id"].split(":")[0] == task)


eval_tasks = [
    "bioasq_list_qa",
    "biored_ner",
    "discomat_te",
    "evidence_inference",
    "multicite_intent_classification",
    "mup_single_document_summarization",
    "qasper_abstractive_qa",
    "scierc_ner",
    "scifact_entailment",
]


for eval_task in eval_tasks:
    func_name = f"filter_{eval_task}"
    globals()[func_name] = partial(filter_task, task=eval_task)
