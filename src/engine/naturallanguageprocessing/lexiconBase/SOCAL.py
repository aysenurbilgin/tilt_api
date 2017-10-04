from csv import reader as csvreader
import os
from collections import OrderedDict
import nltk

""" This file is adapted from https://github.com/Noxeus/opinion-changes-thesis
"""

def get_sentiment(doc):
    return __get_sentiment(doc, 'JJ', 'RB') #best average score according to Ruben's thesis

def __get_sentiment(doc, *tags_used):
    (adjectives,
     adverbs,
     intensifiers,
     nouns,
     verbs) = load_lexicon()
    sentence_polarities = OrderedDict()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    with open(os.path.join(os.path.dirname(__file__), 'stopwords.txt')) as f_open:
        custom_stopwords = set(f_open.read().splitlines())
    stopwords |= custom_stopwords

    for sent_index in doc['sentences']:
        token_scores = []
        sent_as_list = [doc['sentences'][sent_index][
            str(i)] for i in xrange(1, len(doc['sentences'][sent_index])+1)]
        for token_index in doc['sentences'][sent_index]:
            token = doc['sentences'][sent_index][token_index]
            if token in stopwords or len(token) <= 1:
                continue
            pos = doc['pos'][sent_index][token_index]
            lemma = doc['lemmas'][sent_index][token_index]
            # Noun
            if pos == 'NN' and 'NN' in tags_used:
                token_sent = nouns.get(lemma)
                if not token_sent:
                    continue
                token_scores.append(token_sent)
            # Adjective
            elif pos[:2] == 'JJ' and 'JJ' in tags_used:
                token_sent = adjectives.get(token)
                if not token_sent:
                    continue
                multiplier = get_intensity(intensifiers, sent_as_list, token_index)
                token_sent *= multiplier
                if doc['negations'][sent_index].get(token_index):
                    if token_sent > 0:
                        token_sent -= 0.8
                    elif token_sent < 0:
                        token_sent += 0.8
                if token_sent < 0:
                    token_sent *= 1.5
                token_scores.append(token_sent)
            # Adverbs
            elif pos[:2] == 'RB' and 'RB' in tags_used:
                token_sent = adverbs.get(lemma)
                if not token_sent:
                    continue
                multiplier = get_intensity(intensifiers, sent_as_list, token_index)
                token_sent *= multiplier
                if doc['negations'][sent_index].get(token_index):
                    if token_sent > 0:
                        token_sent -= 0.8
                    elif token_sent < 0:
                        token_sent += 0.8
                if token_sent < 0:
                    token_sent *= 1.5
                token_scores.append(token_sent)
            # Verbs
            elif pos[:2] == 'VB' and 'VB' in tags_used:
                token_sent = verbs.get(lemma)
                if not token_sent:
                    continue
                multiplier = get_intensity(intensifiers, sent_as_list, token_index)
                token_sent *= multiplier
                token_scores.append(token_sent)

        #Normalize scores to -1 to 1 range
        token_scores = [1.0*x/5 for x in token_scores]
        devider = lambda x: len(x) if len(x) != 0 else 1
        sentence_polarity = 1.0 * sum(token_scores) / devider(token_scores)
        sentence_polarities[sent_index] = sentence_polarity
    return (1.0 * sum(sentence_polarities.values()) / len(sentence_polarities),
            sentence_polarities)


def get_intensity(intensifiers, sentence, token_index):
    four_ending = set(
        [x.split(' ')[-1] for x in intensifiers.keys() if x.count(' ') == 3])
    three_ending = set(
        [x.split(' ')[-1] for x in intensifiers.keys() if x.count(' ') == 2])
    two_ending = set(
        [x.split(' ')[-1] for x in intensifiers.keys() if x.count(' ') == 1])
    intensity = 1
    search_iter = reversed(xrange(int(token_index)))
    for i in search_iter:
        if i >= 4 and sentence[i] in four_ending:
            possible_int = ' '.join(sentence[i-3:i+1])
            m = intensifiers.get(possible_int)
            if m:
                intensity *= 1 + float(m)
                search_iter.next()
                search_iter.next()
                search_iter.next()
        if i >= 3 and sentence[i] in three_ending:
            possible_int = ' '.join(sentence[i-2:i+1])
            m = intensifiers.get(possible_int)
            if m:
                intensity *= 1 + float(m)
                search_iter.next()
                search_iter.next()
        if i >= 2 and sentence[i] in two_ending:
            possible_int = ' '.join(sentence[i-1:i+1])
            m = intensifiers.get(possible_int)
            if m:
                intensity *= 1 + float(m)
                search_iter.next()
        if i >= 1 and sentence[i] in intensifiers:
            m = intensifiers.get(sentence[i], 0)
            intensity *= 1 + float(m)
        else:
            return intensity
    return intensity


def load_lexicon():
    # Fill lexicon dicts
    with open(os.path.join(os.path.dirname(__file__), 'adj_dictionary1.11.txt'), 'rb') as adjs, \
            open(os.path.join(os.path.dirname(__file__), 'adv_dictionary1.11.txt'), 'rb') as advs, \
            open(os.path.join(os.path.dirname(__file__), 'int_dictionary1.11.txt'), 'rb') as ints, \
            open(os.path.join(os.path.dirname(__file__), 'noun_dictionary1.11.txt'), 'rb') as nns, \
            open(os.path.join(os.path.dirname(__file__), 'verb_dictionary1.11.txt'), 'rb') as vbs:

        reader = csvreader(adjs, delimiter='\t')
        adjectives = {row[0].replace('_', ' '):
                      float(row[1]) for row in reader}

        reader = csvreader(advs, delimiter='\t')
        adverbs = {row[0].replace('_', ' '):
                   float(row[1]) for row in reader}

        reader = csvreader(ints, delimiter='\t')
        intensifiers = {row[0].replace('_', ' '):
                        float(row[1]) for row in reader}

        reader = csvreader(nns, delimiter='\t')
        nouns = {row[0].replace('_', ' '):
                 float(row[1]) for row in reader}

        reader = csvreader(vbs, delimiter='\t')
        verbs = {row[0].replace('_', ' '):
                 float(row[1]) for row in reader}
    return adjectives, adverbs, intensifiers, nouns, verbs
