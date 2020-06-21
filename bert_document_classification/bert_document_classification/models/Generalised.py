from ..document_bert import BertForDocumentClassification
from .util import get_model_path
class Classifier(BertForDocumentClassification):
    def __init__(self, labels, device='cuda', batch_size=10, model_name="doc_lstm"):
        model_path = get_model_path(model_name)

        self.labels = labels
        
        super().__init__(device=device,
                         batch_size=batch_size,
                         bert_batch_size=7,
                         bert_model_path=model_path,
                         architecture='DocumentBertLSTM',
                         labels=self.labels)
        
        