from os.path import join
import os
from glob import glob

from src.engine import EngineConstants

__author__ = 'abilgin'

class CorpusHandler(object):

    def __init__(self, experiment, keyword, similar_keywords_dictionary):

        self.genres = experiment.genre
        self.corpus_list = experiment.corpus_list
        self.similar_keywords_dictionary = similar_keywords_dictionary

        for bin in self.similar_keywords_dictionary.keys():
            self.similar_keywords_dictionary[bin].add(keyword)

        self.unigram_keywords = {}           # dictionary of unigram keywords per time interval
        self.bigram_keywords = {}
        self.trigram_keywords = {}

        # statistics
        self.num_total_selected_sentences_using_seed = 0
        self.num_total_selected_sentences_using_model = 0
        self.freq_mined_keywords_using_seed = {}          # keywords found in the dataset per time interval per genre using the user specified aspects
        self.freq_mined_keywords_using_model = {}         # keywords found in the dataset per time interval per genre using the DSM model

    def _checkUnigramExistence(self, kw, keyword_frequencies, text):
        i = 0
        found = False
        words_of_sentence = text.split(" ")
        while i < len(words_of_sentence):
            if words_of_sentence[i] == kw:
                self._modifyKeywordCounts(kw, keyword_frequencies)
                found = True
            i += 1
        return found

    def _checkBigramExistence(self, kw, keyword_frequencies, text):
        ngram = kw.split(" ")
        i = 0
        found = False
        words_of_sentence = text.split(" ")
        while i < len(words_of_sentence) - 1:
            if words_of_sentence[i] == ngram[0] and words_of_sentence[i + 1] == ngram[1]:
                self._modifyKeywordCounts(kw, keyword_frequencies)
                i += 1
                found = True
            i += 1
        return found

    def _checkTrigramExistence(self, kw, keyword_frequencies, text):
        ngram = kw.split(" ")
        i = 0
        found = False
        words_of_sentence = text.split(" ")
        while i < len(words_of_sentence) - 2:
            if words_of_sentence[i] == ngram[0] and words_of_sentence[i + 1] == ngram[1] and words_of_sentence[i + 2] == ngram[2]:
                self._modifyKeywordCounts(kw, keyword_frequencies)
                i += 2
                found = True
            i += 1
        return found

    def _modifyKeywordCounts(self, kw, keyword_frequencies):
        if kw in keyword_frequencies.keys():
            keyword_frequencies[kw] += 1
        else:
            keyword_frequencies[kw] = 1

    def selectSentencesForSentimentAnalysis(self, requested_corpus_list):
        selected_sentences = {}     # dictionary of dictionary of dictionary - corpus, year and genre and text
        # identify keywords according to ngram
        for time in self.similar_keywords_dictionary.keys():
            self.unigram_keywords[time] = []
            self.bigram_keywords[time] = []
            self.trigram_keywords[time] = []
            # for each time interval retrieve those keywords
            keywords_of_time = self.similar_keywords_dictionary[time]
            for similar in keywords_of_time:
                ngram = similar.split(" ")
                if len(ngram) == 1:
                    self.unigram_keywords[time].append(similar)
                elif len(ngram) == 2:
                    self.bigram_keywords[time].append(similar)
                elif len(ngram) == 3:
                    self.trigram_keywords[time].append(similar)

        for corpus in requested_corpus_list:

            print("Selecting sentences from " + corpus)

            selected_sentences[corpus] = {}
            files = [y for x in os.walk(EngineConstants.ORIGINAL_SOURCE_PATH+corpus+"/") for y in glob(os.path.join(x[0], '*.txt'))]
            for f in files:
                head, tail = os.path.split(f)
                timestamp = tail.split("_")[0]
                genre_we = tail.split("_")[1]
                genre = (genre_we.split(".")[0]).lower()
                # to collect all the news UKnews, worldews etc., combine them
                if "news" in genre and "news" in self.genres:
                    genre = "news"

                if "nonnews" in self.genres and "news" in genre:
                    continue

                binForKeywords = ""
                for timeIndicator in self.similar_keywords_dictionary.keys():
                    endpoints = timeIndicator.split("-")
                    start_year = endpoints[0]
                    end_year = endpoints[1]
                    if timestamp < end_year and timestamp >= start_year:
                        binForKeywords = timeIndicator
                        break
                if binForKeywords == "":
                    continue

                if timestamp not in self.freq_mined_keywords_using_model:
                    self.freq_mined_keywords_using_model[timestamp] = {}
                    self.freq_mined_keywords_using_model[timestamp][genre] = []

                fin = open(join(EngineConstants.ORIGINAL_SOURCE_PATH+corpus+"/", f), 'r')
                all_sentences = fin.read().decode('utf-8', 'ignore').splitlines()
                for sent in all_sentences:
                    uniflag = False
                    biflag = False
                    triflag = False
                    if len(sent) > 3:
                        sent = self.stripSentenceFromNonCharacters(sent)
                        # for each keyword
                        for kw in self.unigram_keywords[binForKeywords]:
                            uniflag = self._checkUnigramExistence(kw, self.freq_mined_keywords_using_model[timestamp], sent)
                            if uniflag:
                                break
                        for kw in self.bigram_keywords[binForKeywords]:
                            biflag = self._checkBigramExistence(kw, self.freq_mined_keywords_using_model[timestamp], sent)
                            if biflag:
                                break
                        for kw in self.trigram_keywords[binForKeywords]:
                            triflag = self._checkTrigramExistence(kw, self.freq_mined_keywords_using_model[timestamp], sent)
                            if triflag:
                                break
                        if uniflag or biflag or triflag:
                            self.num_total_selected_sentences_using_model += 1
                            if timestamp not in selected_sentences[corpus]:
                                selected_sentences[corpus][timestamp] = {}
                                selected_sentences[corpus][timestamp][genre] = []
                            elif genre not in selected_sentences[corpus][timestamp]:
                                selected_sentences[corpus][timestamp][genre] = []

                            selected_sentences[corpus][timestamp][genre].append(sent)
                fin.close()

        print("Selected sentences from " + str(len(selected_sentences.keys())) + " source(s)")
        return selected_sentences

    def stripSentenceFromNonCharacters(self, sentence):
        return ''.join([i if ord(i) < 128 else ' ' for i in sentence])