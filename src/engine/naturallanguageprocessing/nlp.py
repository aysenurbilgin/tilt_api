import traceback
import json
from collections import OrderedDict
from pycorenlp import StanfordCoreNLP
from src.common.database import Database
from src.engine import EngineConstants

DATABASE = Database()

""" This class is adapted from https://github.com/Noxeus/opinion-changes-thesis
"""

class NLPOperator:

    def __init__(self):
        self.corenlp = StanfordCoreNLP('http://localhost:9000')

    def process(self, doc):

        # If already preprocessed
        if set(doc.keys()) == {'texthash', 'date', 'genre', 'source', 'exp_id', 'num_neighbours',
                               'original_text', 'sentiment', 'keyword', 'sentence_polarities',
                               'sentences', 'pos', 'negations', 'lemmas', '_id', 'ner'}:

            return
        # fix some encoding issues in the doc text
        try:
            text = doc['original_text'].encode('utf-8'). \
                replace('\x05', ''). \
                replace('\x12', ''). \
                replace('\x13', ''). \
                replace('\x14', ''). \
                replace('\x16', ''). \
                replace(' @', ' '). \
                replace('//', ''). \
                replace('<p> ', '')
            if len(text) <= 100000:
                sentences, pos, lemmas, ner, negs = self._pre_process_doc(text)
                if pos and lemmas and ner and negs:
                    DATABASE.update(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'_id': doc['_id']},
                                    {'$set': {
                                 'sentences': sentences,
                                 'pos': pos,
                                 'lemmas': lemmas,
                                 'ner': ner,
                                 'negations': negs
                             }
                    })
                else:
                    print ("Failed on " + doc['_id'] + " for reason " + sentences)
            else:
                sentences = OrderedDict()
                pos = OrderedDict()
                lemmas = OrderedDict()
                ner = OrderedDict()
                negs = OrderedDict()

                # split_count = len(text) // 100000 + (len(text) % 100000)
                split_count = len(text) // 100000 + 1
                offset = 0
                for run_time in range(split_count):
                    start = run_time * 100000
                    end = (run_time + 1) * 100000
                    chunk = text[start:end]
                    (sentences_, pos_, lemmas_, ner_, negs_) = self._pre_process_doc(chunk, offset)
                    if not sentences_:
                        print("Sentences partial empty!")
                        break
                    sentences.update(sentences_)
                    pos.update(pos_)
                    lemmas.update(lemmas_)
                    ner.update(ner_)
                    negs.update(negs_)

                    offset = int(sentences_.keys()[-1]) + 1

                DATABASE.update(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'_id': doc['_id']},
                                {'$set': {
                                    'sentences': sentences,
                                    'pos': pos,
                                    'lemmas': lemmas,
                                    'ner': ner,
                                    'negations': negs
                                }
                })
        except:
            traceback.print_exc()

    def _pre_process_doc(self, text, offset=0):
        """ Annotate a document using StanfordNLP"""
        properties = {
            'annotators': 'lemma, pos, ner, depparse',
            'outputFormat': 'json',
            'timeout': 120000
        }
        output = self.corenlp.annotate(text, properties=properties)
        sentences = OrderedDict()
        pos = OrderedDict()
        lemmas = OrderedDict()
        ner = OrderedDict()
        negations = OrderedDict()
        try:
            if not isinstance(output, dict):
                output = json.loads(output, encoding='utf-8', strict=False)
            assert isinstance(output, dict)
            for sentence in output['sentences']:
                sent_index = str(sentence['index'] + offset)
                sentences[sent_index] = OrderedDict()
                pos[sent_index] = OrderedDict()
                lemmas[sent_index] = OrderedDict()
                ner[sent_index] = OrderedDict()
                negations[sent_index] = OrderedDict()
                tokens = sentence['tokens']
                for token in tokens:
                    token_index = str(token['index'])
                    sentences[sent_index][token_index] = token['originalText']
                    pos[sent_index][token_index] = token['pos']
                    lemmas[sent_index][token_index] = token['lemma']
                    if token['ner'] != "O":
                        ner[sent_index][token_index] = token['ner']
                # for dep in sentence['basic-dependencies']:
                for dep in sentence['basicDependencies']:
                    if dep['dep'] == 'neg':
                        negations[sent_index][str(dep['governor'])] = True

            return (sentences, pos, lemmas, ner, negations)
        except AssertionError:
            traceback.print_exc()
            return (output, None, None, None, None)
        except:
            traceback.print_exc()
            return (output, None, None, None, None)
