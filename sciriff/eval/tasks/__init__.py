from .bioasq_list_qa import BioASQ
from .biored_ner import Biored
from .evidence_inference import EvidenceInference
from .multicite_intent_classification import MultiCite
from .mup_single_document_summarization import MUP
from .qasper_abstractive_qa import Qasper
from .scierc_ner import SciERC
from .scifact_entailment import SciFact

registry = {
    "bioasq_list_qa": BioASQ,
    "biored_ner": Biored,
    "evidence_inference": EvidenceInference,
    "multicite_intent_classification": MultiCite,
    "mup_single_document_summarization": MUP,
    "qasper_abstractive_qa": Qasper,
    "scierc_ner": SciERC,
    "scifact_entailment": SciFact
}
