import nltk
import os
from nltk.corpus import sentiwordnet as swn
from collections import OrderedDict

""" This file is adapted from https://github.com/Noxeus/opinion-changes-thesis
"""
def get_sentiment(doc):
    return __calc_sentiment(doc, 'JJ', 'RB', 'VB') #best average score according to Ruben's thesis

def __calc_sentiment(doc, *tags_used):
    sentences = doc['sentences']
    pos_tags = doc['pos']
    lemmas = doc['lemmas']
    negations = doc['negations']
    sentence_polarities = OrderedDict()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    with open(os.path.join(os.path.dirname(__file__), "stopwords.txt")) as f_open:
        custom_stopwords = set(f_open.read().splitlines())
    stopwords |= custom_stopwords

    for sent_index, sentence_tokens in sentences.iteritems():
        poss = []
        negs = []
        for token_index, token in sentence_tokens.iteritems():
            if token in stopwords or len(token) <= 1:
                continue

            pos = pos_tags[sent_index][token_index]
            if not pos[:2] in tags_used or pos == 'NNP':
                continue
            conv_tags = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r', 'NNS': 'n', 'JJS': 'a'}
            synsets = swn.senti_synsets(token, pos=conv_tags[pos[:2]])
            for synset in synsets:
                # Invert score if negated word
                if negations[sent_index].get(token_index):
                    pol = synset.pos_score()-synset.neg_score()
                    if pol > 0:
                        poss.append(synset.pos_score()-0.8)
                        negs.append(synset.neg_score()+0.8)
                    elif pol < 0:
                        poss.append(synset.pos_score()+0.8)
                        negs.append(synset.neg_score()-0.8)
                    else:
                        poss.append(synset.pos_score())
                        negs.append(synset.neg_score())
                else:
                    poss.append(synset.pos_score())
                    negs.append(synset.neg_score())
        devider = lambda x: len(x) if len(x) != 0 else 1
        # Negative weighting
        # negs = [x*1.5 for x in negs]
        sentence_polarity = (1.0 * sum(poss) / devider(poss)) - (1.0 * sum(negs) / devider(negs))
        sentence_polarities[sent_index] = sentence_polarity
    return (1.0 * sum(sentence_polarities.values()) / len(sentence_polarities),
            sentence_polarities)