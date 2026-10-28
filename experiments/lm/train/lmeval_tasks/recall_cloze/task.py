
import ast
import re
from typing import List
import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

def contains_score(prediction: str, labels: List[str]):
    return max((int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction))) for label in labels))

def _answers(doc) -> List[str]:
    a = doc['answers']
    if isinstance(a, str):
        a = ast.literal_eval(a)
    return [str(x) for x in a] or ['unanswerable']

class _ClozeRecall(ConfigurableTask):
    VERSION = 0
    DATASET_NAME = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset['validation']

    def doc_to_target(self, doc):
        return ' ' + _answers(doc)[0]

    def construct_requests(self, doc, ctx, **kwargs):
        return [Instance(request_type='generate_until', doc=doc, arguments=(ctx, {'until': ['\n'], 'max_gen_toks': 48}), idx=0, **kwargs)]

    def process_results(self, doc, results):
        return {'contains': contains_score(results[0], _answers(doc))}

    def aggregation(self):
        return {'contains': np.mean}

    def higher_is_better(self):
        return {'contains': True}

class BasedTriviaQA(_ClozeRecall):
    DATASET_PATH = 'hazyresearch/based_triviaqa'

    def doc_to_text(self, doc):
        context = re.sub('\\[PAR\\]|\\[DOC\\]|\\[TLE\\]', '', doc['context'].strip())
        question = doc['question'].strip()
        while context.lower().endswith(question.lower()):
            context = context[:-len(question)]
        return context.strip().strip('.') + '. ' + doc['question'].strip()

class BasedNQ1024(_ClozeRecall):
    DATASET_PATH = 'hazyresearch/based_nq_1024'

    def doc_to_text(self, doc):
        context = doc['context'].strip()
        question = doc['question'].strip()
        if context.endswith(question):
            context = context[:-len(question)]
        return context.strip().strip('.') + '. ' + doc['question'].strip()

class BasedDrop(_ClozeRecall):
    DATASET_PATH = 'hazyresearch/based_drop'

    def doc_to_text(self, doc):
        context = doc['context'].strip()
        question = doc['question'].strip()
        while context.lower().endswith(question.lower()):
            context = context[:-len(question)]
        return context.strip().strip('.') + '. ' + question
