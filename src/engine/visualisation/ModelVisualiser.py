from __future__ import division
import pickle
from collections import defaultdict

import datetime
import time

import xxhash
from bokeh.embed import components

from src.common.database import Database
from src.engine import EngineConstants
from src.engine.EngineConstants import EngineUtils
from src.engine.corpusprocessing.CorpusHandler import CorpusHandler
from src.engine.naturallanguageprocessing.LexiconBasedSA import LexiconBasedSentimentAnalyser

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import collections
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from bokeh.layouts import column
from bokeh.plotting import figure

DATABASE = Database()
EU = EngineUtils()

__author__ = 'abilgin'

class ModelVisualiser:

    def __init__(self, experiment, keyword, num_neighbours):

        self.experiment = experiment
        self.keyword = keyword
        self.number_of_neighbours = int(num_neighbours)

        self.bins = sorted(self.experiment.existing_models.keys())

        self.neighbours = set()
        self.neighbours_per_bin = {}
        self.neighbours_per_bin_with_score = {}

        self.keyword_vector_per_bin = {}

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.logger = EU.setup_logger(__name__, EngineConstants.EXP_LOG_PATH + self.experiment._id + timestr + '.log')

        self.keyword_error = EngineConstants.NO_KEYERR
        self._analyseVSModelSimilarity()

    def _analyseVSModelSimilarity(self):

        count_error = 0

        for bin in self.bins:
            # initialise the data structures
            self.neighbours_per_bin[bin] = set()
            self.neighbours_per_bin_with_score[bin] = []

            # Load a model from db
            pickled_model = DATABASE.getGridFS().get(self.experiment.existing_models[bin]).read()
            model = pickle.loads(pickled_model)

            # get the word vector of the keyword for bin
            keyword_bin_id = self.keyword + "(" + str(bin) + ")"
            if keyword_bin_id not in self.keyword_vector_per_bin.keys():
                try:
                    self.keyword_vector_per_bin[keyword_bin_id] = model.wv[self.keyword]
                except:
                    count_error += 1
                    self.logger.error(str(bin) + ": Vector could not be retrieved for " + self.keyword)

            # get the nearest neighours for keyword
            if self.number_of_neighbours > 0:
                similar_neighbours = self._queryModelForKeyword(model)
                if len(similar_neighbours) > 0:
                    for similar_word in similar_neighbours:
                        try:
                            self.neighbours_per_bin[bin].add(similar_word[0])
                            self.neighbours_per_bin_with_score[bin].append(similar_word)
                        except:
                            self.logger.error("Similarity analysis error for " + self.keyword + " in " + bin)

                        self.neighbours.add(similar_word[0])

        if count_error == len(self.bins):
            self.keyword_error = EngineConstants.ALL_KEYERR
        elif count_error > 0:
            self.keyword_error = EngineConstants.PARTIAL_KEYERR

    def _queryModelForKeyword(self, model):

        word_list= []
        lem = WordNetLemmatizer()
        try:
            word_list_raw = model.wv.most_similar(positive = [self.keyword], negative = [], topn = self.number_of_neighbours)

            for word in word_list_raw:
                if not set('[~!@#$%^&*()_+{}":;\'`]+$').intersection(str(word[0])):
                    word_lem = lem.lemmatize(str(word[0]))
                    if word_lem not in word_list:
                        word_list.append([word_lem, word[1]])

            words = [str(v[0]) for v in word_list]

            if not words:
                self.logger.info('Keyword ' + self.keyword + ' not found with cosine!')
            else:
                self.logger.info("Most similar words: " + ",".join(words))
        except Exception as e:
            self.logger.exception(e)

        # TODO: Return the agreement set of the different similarity measures
        return word_list

    def distanceBasedAspectVisualisation(self, aspect_words):
        modern_model = self._retrieveMostRecentModel()
        return self._drawSimilarityDistancePlot(modern_model, aspect_words)

    def _retrieveMostRecentModel(self):
        # the most recent model should be the last one
        final_existing_bin = self.bins[-1]
        pickled_model = DATABASE.getGridFS().get(self.experiment.existing_models[final_existing_bin]).read()
        recent_model = pickle.loads(pickled_model)
        return recent_model

    def _drawSimilarityDistancePlot(self, modern_model, aspect_words):

        s = []
        words_not_found = []
        # sort the bins
        self.neighbours_per_bin = collections.OrderedDict(sorted(self.neighbours_per_bin.items()))
        # for each aspect word
        for aspect in aspect_words:
            mean_scores_of_neighbours_per_bin = {}
            similarity_scores_of_keyword_per_bin = {}
            for bin in self.bins:
                similarity_scores_of_keyword_per_bin[bin], display_key_label = self._calculateBinSimilarityWithAspect(modern_model, bin, aspect)
                bin_total = 0
                for neighbour in self.neighbours_per_bin[bin]:
                    # retrieve similarity with sentiment and take average
                    try:
                        bin_total += float(modern_model.wv.similarity(neighbour, aspect))
                    except:
                        bin_total += 0

                if len(self.neighbours_per_bin[bin]) > 0:
                    mean_scores_of_neighbours_per_bin[bin] = bin_total / float(len(self.neighbours_per_bin[bin]))
                else:
                    mean_scores_of_neighbours_per_bin[bin] = 0


            if np.any(np.array(similarity_scores_of_keyword_per_bin.values())) or np.any(np.array(similarity_scores_of_keyword_per_bin.values())):

                similarity_scores_of_keyword_per_bin = collections.OrderedDict(sorted(similarity_scores_of_keyword_per_bin.items()))
                key_sims = np.array(similarity_scores_of_keyword_per_bin.values())

                mean_scores_of_neighbours_per_bin = collections.OrderedDict(sorted(mean_scores_of_neighbours_per_bin.items()))
                means = np.array(mean_scores_of_neighbours_per_bin.values())

                fig = figure(x_range=self.bins, width=800, plot_height=300, title="'" + aspect + "'")

                fig.xaxis.axis_label = "Time Intervals"
                fig.yaxis.axis_label = "Similarity"
                fig.yaxis.major_label_orientation = "vertical"
                fig.yaxis.bounds = [0,1]

                fig.axis.minor_tick_in = -3
                fig.axis.axis_line_width = 3

                fig.line(self.bins, key_sims.tolist(), legend=self.keyword, line_color="firebrick", line_width=4)
                fig.line(self.bins, means.tolist(), legend="Mean of neighbours", line_color="navy", line_width=4, line_dash=[4, 4])
                fig.legend.background_fill_alpha = 0.5

                s.append(fig)

            else:
                words_not_found.append(aspect)

        # put all the plots in a column
        list = [s[i] for i in range(0, len(s))]
        p = column(list)

        script, div = components(p)
        return script, div, words_not_found

    def _calculateBinSimilarityWithAspect(self, model, bin, aspect):

        for key_bin_id in self.keyword_vector_per_bin.keys():
            if bin in key_bin_id:
                vec_key = self.keyword_vector_per_bin[key_bin_id]
                try:
                    return np.dot(vec_key, model.wv[aspect])/(np.linalg.norm(vec_key)* np.linalg.norm(model.wv[aspect])), key_bin_id
                except:
                    return 0, ""

        return 0, ""

    def timeTrackingVisualisation(self, aspects, algorithm, tsne_perp, tsne_iter):
        modern_model = self._retrieveMostRecentModel()
        return self._drawTrackingPlot(modern_model, aspects, algorithm, tsne_perp, tsne_iter)

    def _drawTrackingPlot(self, modern_model, aspects, algorithm, tsne_perp, tsne_iter):
        # Prepare figure
        fig = figure(width=1200, plot_height=600, title="Semantic time travel of '" + self.keyword + "' using " + algorithm)

        if self.neighbours:
            # find the union of keyword's k nearest neighbours over all time points
            surrounding_words = filter(bool, self.neighbours)

            if aspects:
                surrounding_words.extend(aspects)

            embeddings = self._getEmbeddingsFromModelForWords(modern_model, surrounding_words)

            for key, value in self.keyword_vector_per_bin.items():
                embeddings[key] = value

            vectors = embeddings.values()
            words = embeddings.keys()

            if algorithm == "pca":
                pca = PCA(n_components=2, whiten=True)
                vectors2d = pca.fit(vectors).transform(vectors)
            else:
                # perplexity ranges from 20 to 50
                tsne = TSNE(perplexity=tsne_perp, n_components=2, init='pca', n_iter=tsne_iter, method='exact')
                vectors2d = tsne.fit_transform(vectors)

            bin_keyword_vectors_x = []
            bin_keyword_vectors_y = []
            bin_words = []

            default_neighbour_vectors_x = []
            default_neighbour_vectors_y = []
            default_neighbour_words = []

            aspect_vectors_x = []
            aspect_vectors_y = []
            aspect_words = []

            for point, word in zip(vectors2d, words):
                # categorise points
                if "(" in word:
                    bin_keyword_vectors_x.append(point[0])
                    bin_keyword_vectors_y.append(point[1])
                    bin_words.append(word)
                elif word in aspects:
                    aspect_vectors_x.append(point[0])
                    aspect_vectors_y.append(point[1])
                    aspect_words.append(word)
                else:
                    default_neighbour_vectors_x.append(point[0])
                    default_neighbour_vectors_y.append(point[1])
                    default_neighbour_words.append(word)

            fig.circle(default_neighbour_vectors_x, default_neighbour_vectors_y,
                      line_color="black", fill_color="blue", fill_alpha=0.5, size=10)
            fig.text(default_neighbour_vectors_x, default_neighbour_vectors_y, default_neighbour_words, text_font_size="10pt")

            fig.square(aspect_vectors_x, aspect_vectors_y,
                       line_color="black", fill_color="black", fill_alpha=0.5, size=15)
            fig.text(aspect_vectors_x, aspect_vectors_y, aspect_words, text_font_size="15pt")

            fig.triangle(bin_keyword_vectors_x, bin_keyword_vectors_y,
                       line_color="black", fill_color="red", fill_alpha=0.5, size=12)
            fig.text(bin_keyword_vectors_x, bin_keyword_vectors_y, bin_words, text_font_size="12pt")

        script, div = components(fig)
        return script, div

    def _getEmbeddingsFromModelForWords(self, model, word_list):

        similar_embeddings = {}

        for word in word_list:
            try:
                similar_embeddings[word] = model.wv[word]
            except:
                self.logger.info(word + " not found in model.")

        return similar_embeddings

    def sentimentVisualisation(self, lexicon, requested_corpus_list):

        if not requested_corpus_list:
            requested_corpus_list = self.experiment.corpus_list
        # check for existing keyword and corpus
        missing_corpora = []
        for corpus in requested_corpus_list:
            res = DATABASE.find_one(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'exp_id': self.experiment._id, 'keyword': self.keyword,
                                                             'num_neighbours': self.number_of_neighbours, 'source':corpus})
            if not res:
                missing_corpora.append(corpus)

        if len(missing_corpora) > 0:
            ch = CorpusHandler(self.experiment, self.keyword, self.neighbours_per_bin)
            selected_sentences = ch.selectSentencesForSentimentAnalysis(missing_corpora)
            self._uploadToDB(selected_sentences)
        sentiment_analyser = LexiconBasedSentimentAnalyser(self.experiment._id, self.keyword, self.number_of_neighbours)
        sentiment_analyser.fromDBRunAnalysis()

        return self._drawSentimentEvolutionPlot(lexicon, requested_corpus_list)


    def _uploadToDB(self, selected_sentences):
        for corpus in selected_sentences.keys():
            for time in selected_sentences[corpus].keys():
                for genre in selected_sentences[corpus][time].keys():
                    text = ". ".join(selected_sentences[corpus][time][genre]).encode('utf-8')
                    texthash = xxhash.xxh64(text).hexdigest()
                    DATABASE.update(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'texthash': texthash},
                                    {'date': time, 'source': corpus, 'genre': genre, 'exp_id': self.experiment._id,
                                     'keyword' : self.keyword, 'num_neighbours' : self.number_of_neighbours,
                                     'original_text': text, 'texthash': texthash, 'sentiment': {}, 'sentence_polarities': {}})

    def _drawSentimentEvolutionPlot(self, lexicon, corpus_list):

        data = self._get_plotdata(lexicon, corpus_list)

        if not len(data):
            self.keyword_error = EngineConstants.NO_DATA

        fig = figure(width=1200, plot_height=600, title="Sentiment analysis of '" + self.keyword + "' using " + lexicon + " within " + ', '.join(corpus_list))
        fig.square(data.keys(), data.values(), line_color="black", fill_color="blue", fill_alpha=0.5, size=10)

        fig.xaxis.axis_label = "Time"
        fig.yaxis.axis_label = "Sentiment Orientation"
        fig.axis.minor_tick_in = -3
        fig.axis.axis_line_width = 3

        script, div = components(fig)
        return script, div

    def _get_plotdata(self, lexicon, corpus_list):

        sentiment_per_timestamp = defaultdict(list)
        no_date_counter = 0
        use_dates = False

        for corpus in corpus_list:
            for doc in DATABASE.iter_collection(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'exp_id': self.experiment._id,
                                                                             'num_neighbours': self.number_of_neighbours,
                                                                             'keyword': self.keyword, 'source': corpus,
                                                                             'sentiment.' + lexicon: {'$exists': True}}):
                if "date" in doc and doc['date']:
                    use_dates = True
                    if isinstance(doc['date'], int):
                        date = doc['date']
                    elif not isinstance(doc['date'], datetime.datetime):
                        date = int(doc['date'][:4])
                    else:
                        date = doc['date'].year
                elif doc['sentiment'] and not use_dates:
                    date = no_date_counter
                    no_date_counter += 1

                sentiment_per_timestamp[date].append(doc['sentiment'][lexicon])

        for time, sent_group in sentiment_per_timestamp.items():
            if len(sent_group) > 1:
                # take the average of genres
                genre_sum = 0
                for sent_genre in sent_group:
                    genre_sum += sent_genre
                average_sent = genre_sum / len(sent_group)
                sentiment_per_timestamp[time] = [average_sent]

        if use_dates:
            for date in range(min(sentiment_per_timestamp.keys()), max(sentiment_per_timestamp.keys())):
                if not date in sentiment_per_timestamp:
                    sentiment_per_timestamp[date] = []

        data = collections.OrderedDict(sorted(sentiment_per_timestamp.items(), key=lambda t: t[0]))
        return data

    def getKeywordErrorStatus(self):
        return self.keyword_error

    def getSimilarKeywordsPerBin(self):
        return self.neighbours_per_bin


