import nltk
import os
from collections import OrderedDict

""" This file is adapted from https://github.com/Noxeus/opinion-changes-thesis
"""

pos_conv = {
    'noun': 'NOUN',
    'verb': 'VERB',
    'adj': 'ADJ',
    'adverb': 'ADV',
    'anypos': 'ANYPOS',
}
sent_conv = {
    ('weaksubj', 'positive'): 0.5,
    ('weaksubj', 'neutral'): 0,
    ('weaksubj', 'negative'): -0.5,
    ('weaksubj', 'weakneg'): -0.25,
    ('weaksubj', 'both'): 0,
    ('strongsubj', 'positive'): 1,
    ('strongsubj', 'neutral'): 0,
    ('strongsubj', 'negative'): -1,
    ('strongsubj', 'both'): 0
}
with open(os.path.join(os.path.dirname(__file__), 'subjclues.ttf')) as fo:
    sub_clues = [c.strip().replace('type=', 'rel=').split() for c in fo]
sub_clues = [
    dict([d.split('=') for d in sc if len(d.split('=')) == 2])
    for sc in sub_clues]
lexicon = {(s['word1'], pos_conv[s['pos1']]):
           (s['rel'],
            s['priorpolarity'],
            True if s['stemmed1'] == 'y' else False
            )
           for s in sub_clues if s['priorpolarity']
           }

def get_sentiment(doc):
    return __get_sentiment(doc, "RB", "JJ") #best average score according to Ruben's thesis

def __get_sentiment(doc, *tags_used):

    try:
        sentences = doc['sentences']
        pos_tags = doc['pos']
        lemmas = doc['lemmas']
        negations = doc['negations']
    except KeyError:
        return None, None
    sentence_polarities = OrderedDict()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    with open(os.path.join(os.path.dirname(__file__), "stopwords.txt")) as f_open:
        custom_stopwords = set(f_open.read().splitlines())
    stopwords |= custom_stopwords
    for sent_index, sentence_tokens in sentences.iteritems():
        token_scores = []
        for token_index, token in sentence_tokens.iteritems():
            if token in stopwords or len(token) <= 1:
                continue

            pos = pos_tags[sent_index][token_index]
            lemma = lemmas[sent_index][token_index]
            if 'JJ' in pos and 'JJ' in tags_used:
                l_entry = lexicon.get((token, 'ADJ'),
                                      lexicon.get((token, 'ANYPOS'))
                                      )

            elif 'RB' in pos and 'RB' in tags_used:
                l_entry = lexicon.get((token, 'ADV'),
                                      lexicon.get((token, 'ANYPOS'))
                                      )
            elif 'NN' in pos and 'NN' in tags_used:
                l_entry = lexicon.get((token, 'NOUN'),
                                         lexicon.get((token, 'ANYPOS'))
                                         )

            elif 'VB' in pos and 'VB' in tags_used:
                l_entry = lexicon.get((token, 'VERB'),
                                         lexicon.get((token, 'ANYPOS'))
                                         )
            else:
                continue
            if l_entry:
                token_sent = sent_conv[l_entry[:2]]
                if token_sent:
                    if negations[sent_index].get(token_index):
                        if token_sent > 0:
                            token_sent -= 0.8
                        elif token_sent < 0:
                            token_sent += 0.8
                    if token_sent < 0:
                        token_sent *= 1.5

                    token_scores.append(token_sent)

        devider = lambda x: len(x) if len(x) != 0 else 1
        sentence_polarity = 1.0 * sum(token_scores) / devider(token_scores)
        sentence_polarities[sent_index] = sentence_polarity
    return (1.0 * sum(sentence_polarities.values()) / len(sentence_polarities),
            sentence_polarities)
