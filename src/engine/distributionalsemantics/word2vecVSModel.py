from __future__ import division

from collections import OrderedDict
import dill as pickle
from gensim.models import word2vec
from gensim.models.phrases import Phrases, Phraser
import os

from src.common.database import Database
from src.engine.EngineConstants import EngineUtils
from src.engine import EngineConstants
import src.engine.distributionalsemantics.ModelConstants as ModelConstants
from src.engine.distributionalsemantics.ModelConstants import ModelAlignment
from src.engine.distributionalsemantics.ModelConstants import ModelSimilarity
from src.models.configurations.configuration import Configuration

DATABASE = Database()
EU = EngineUtils()

class MySentences(object):
    def __init__(self, dirname, genres):
        self.dirname = dirname
        self.genres = genres

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            genre_we = fname.split("_")[1]
            genre = genre_we.split(".")[0]
            # to collect all the news UKnews, worldews, etc., combine them
            if "news" in genre and "news" in self.genres:
                genre = "news"

            if fname.endswith(".txt") and ((genre in self.genres or "all" in self.genres) or ("nonnews" in self.genres and "news" not in genre)):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()

class MySentencesSoFar(object):
    def __init__(self, datafolder, bin, genres, reverse_flag):
        self.datafolder = datafolder
        self.genres = genres

        dirs = os.listdir(datafolder)
        dirs.sort(reverse=reverse_flag)

        for index in range(0,len(dirs)):
            if ".DS" in dirs[index]:
                del dirs[index]
                break

        self.bins_so_far = []

        for dir in dirs:
            if (not reverse_flag and dir <= bin) or (reverse_flag and dir >= bin):
                self.bins_so_far.append(dir)

    def __iter__(self):
        for bin in self.bins_so_far:
            fullpath = self.datafolder + bin + "/"
            for fname in os.listdir(fullpath):
                genre_we = fname.split("_")[1]
                genre = genre_we.split(".")[0]
                # to collect all the news UKnews, worldews, etc., combine them
                if "news" in genre and "news" in self.genres:
                    genre = "news"

                if fname.endswith(".txt") and ((genre in self.genres or "all" in self.genres) or ("nonnews" in self.genres and "news" not in genre)):
                    for line in open(os.path.join(fullpath, fname)):
                        yield line.split()

class word2vecVSModel(Configuration):

    def __init__(self, user_email, **kwargs):

        if 'configuration' in kwargs:
            # creation from the web
            Configuration.__init__(self, user_email=user_email, **kwargs)
            self.existing_models = {}       # key: bin, value: pickled model
            self.analogy_accuracies = {}
            self.similarity_accuracies = {}
            self.all_similar_keywords_per_bin_per_keyword = {}

            if len(self.corpus_list) > 1:
                corpus_identifier = "_".join(self.corpus_list)
            else:
                corpus_identifier = self.corpus_list[0]

            if self.init_flag:
                self.processed_text_dir = os.path.join(EngineConstants.TRAINING_PATH, 'init'+str(self.bin_interval)+'/'+corpus_identifier+'/')
            else:
                self.processed_text_dir = os.path.join(EngineConstants.TRAINING_PATH, str(self.bin_interval)+'/'+corpus_identifier+'/')
        else:
            # default constructor from the database
            self.__dict__.update(kwargs)
            self.user_email = user_email


    def createVSModel(self, until_flag, reverse_flag, online_flag):

        logger = EU.setup_logger(__name__, EngineConstants.EXP_LOG_PATH + self._id + '.log')

        # read directories and files for processed data
        dirs = os.listdir(self.processed_text_dir)
        dirs.sort(reverse=reverse_flag)

        if len(dirs) == 0:
            return

        for index in range(0,len(dirs)):
            if ".DS" in dirs[index]:
                del dirs[index]
                break

        if not bool(self.existing_models):
            self.existing_models = OrderedDict(sorted(self.existing_models.items(), reverse=reverse_flag))

        for bin in dirs:
            if until_flag:
                sentences = MySentencesSoFar(self.processed_text_dir, bin, self.genre, reverse_flag)
            else:
                sentences = MySentences(self.processed_text_dir + bin + "/", self.genre)

            if online_flag and bool(self.existing_models):
                model = self._trainOnline(sentences, bin, reverse_flag, logger)
            else:
                model = self._train(sentences, bin, logger)

            self.existing_models[bin] = DATABASE.getGridFS().put(pickle.dumps(model))
            self._evaluateWordAnalogy(bin, model, logger)
            self._evaluateWordSimilarity(bin, model, logger)
            self.save_to_db()


    def createAlignedVSModel(self, until_flag, reverse_flag, online_flag):

        logger = EU.setup_logger(__name__, EngineConstants.EXP_LOG_PATH + self._id + '.log')

        # read directories and files for processed data
        dirs = os.listdir(self.processed_text_dir)
        dirs.sort(reverse=reverse_flag)

        if len(dirs) == 0:
            return

        for index in range(0,len(dirs)):
            if ".DS" in dirs[index]:
                del dirs[index]
                break

        if not bool(self.existing_models):
            # take the first bin as a base, to be aligned with the rest
            sentences = MySentences(self.processed_text_dir + dirs[0] + "/", self.genre)
            base_embed = self._train(sentences, dirs[0], logger)
            self.existing_models[dirs[0]] = DATABASE.getGridFS().put(pickle.dumps(base_embed))
            self._evaluateWordAnalogy(dirs[0], base_embed, logger)
            self._evaluateWordSimilarity(dirs[0], base_embed, logger)
            self.save_to_db()

            for bin in dirs:
                if bin != dirs[0]:
                    if until_flag:
                        sentences = MySentencesSoFar(self.processed_text_dir, bin, self.genre, reverse_flag)
                    else:
                        sentences = MySentences(self.processed_text_dir + bin + "/", self.genre)

                    if online_flag and bool(self.existing_models):
                        new_embed = self._trainOnline(sentences, bin, reverse_flag, logger)

                    else:
                        new_embed = self._train(sentences, bin, logger)

                    ma = ModelAlignment()
                    aligned_embed = ma.smart_procrustes_align_gensim(base_embed, new_embed)
                    self.existing_models[bin] = DATABASE.getGridFS().put(pickle.dumps(aligned_embed))
                    self._evaluateWordAnalogy(bin, aligned_embed, logger)
                    self._evaluateWordSimilarity(bin, aligned_embed, logger)
                    self.save_to_db()

                    base_embed = aligned_embed

        else:
            self.existing_models = OrderedDict(sorted(self.existing_models.items(), reverse=reverse_flag))
            # get the final existing model to make it base model
            final_existing_bin = self.existing_models.keys()[-1]
            sentences = MySentences(self.processed_text_dir + final_existing_bin + "/", self.genre)
            base_embed = self._train(sentences, final_existing_bin, logger)
            self.existing_models[final_existing_bin] = DATABASE.getGridFS().put(pickle.dumps(base_embed))
            self._evaluateWordAnalogy(final_existing_bin, base_embed, logger)
            self._evaluateWordSimilarity(final_existing_bin, base_embed, logger)
            self.save_to_db()


            for bin in dirs:
                # pass until the final existing bin
                if (not reverse_flag and bin > final_existing_bin) or (reverse_flag and bin < final_existing_bin):
                    if until_flag:
                        sentences = MySentencesSoFar(self.processed_text_dir, bin, self.genre, reverse_flag)
                    else:
                        sentences = MySentences(self.processed_text_dir + bin + "/", self.genre)

                    if online_flag and bool(self.existing_models):
                        new_embed = self._trainOnline(sentences, bin, reverse_flag, logger)
                    else:
                        new_embed = self._train(sentences, bin, logger)


                    ma = ModelAlignment()
                    aligned_embed = ma.smart_procrustes_align_gensim(base_embed, new_embed)
                    self.existing_models[bin] = DATABASE.getGridFS().put(pickle.dumps(aligned_embed))
                    self._evaluateWordAnalogy(bin, aligned_embed, logger)
                    self._evaluateWordSimilarity(bin, aligned_embed, logger)
                    self.save_to_db()

                    base_embed = aligned_embed


    def _train(self, sentences, bin, logger):

        logger.info("Training " + bin)

        training_algorithm_major = 0
        if "SG" in self.training_algorithm:
            training_algorithm_major = 1

        training_algorithm_minor = 0
        if "HS" in self.training_algorithm:
            training_algorithm_minor = 1

        # Train the word2vec models
        '''min_count ignore all words and bigrams with total collected count lower than this. By default it value is 5.
        threshold represents a threshold for forming the phrases (higher means fewer phrases).
        A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold,
        where N is the total vocabulary size. By default it value is 10.0'''
        if "bi" in self.ngram:
            bigram = Phrases(sentences, min_count=10, threshold=10, delimiter=b' ')
            bigram_phraser = Phraser(bigram)
            model = word2vec.Word2Vec(bigram_phraser[sentences], size=self.dimensions,
                                      # workers=max(1, multiprocessing.cpu_count()-1),
                                      workers=2,#essence
                                      sg=training_algorithm_major, hs=training_algorithm_minor,
                                      window=self.window_size, iter=self.iter,
                                      sample=self.sample, min_count=self.min_count, negative=self.negative,
                                      alpha=ModelConstants.ALPHA, min_alpha=ModelConstants.MIN_ALPHA, cbow_mean=ModelConstants.CBOW_MEAN)
        elif "tri" in self.ngram:
            bigram = Phrases(sentences, min_count=10, threshold=10, delimiter=b' ')
            bigram_phraser = Phraser(bigram)
            trigram = Phrases(bigram_phraser[sentences], min_count=10, threshold=10, delimiter=b' ')
            trigram_phraser = Phraser(trigram)
            model = word2vec.Word2Vec(trigram_phraser[bigram_phraser[sentences]], size=self.dimensions,
                                      # workers=max(1, multiprocessing.cpu_count()-1),
                                      workers=2,#essence
                                      sg=training_algorithm_major, hs=training_algorithm_minor,
                                      window=self.window_size, iter=self.iter,
                                      sample=self.sample, min_count=self.min_count, negative=self.negative,
                                      alpha=ModelConstants.ALPHA, min_alpha=ModelConstants.MIN_ALPHA, cbow_mean=ModelConstants.CBOW_MEAN)
        else:
            model = word2vec.Word2Vec(sentences, size=self.dimensions,
                                      # workers=max(1, multiprocessing.cpu_count()-1),
                                      workers=2,#essence
                                      sg=training_algorithm_major, hs=training_algorithm_minor,
                                      window=self.window_size, iter=self.iter,
                                      sample=self.sample, min_count=self.min_count, negative=self.negative,
                                      alpha=ModelConstants.ALPHA, min_alpha=ModelConstants.MIN_ALPHA, cbow_mean=ModelConstants.CBOW_MEAN)

        return model


    def _trainOnline(self, new_sentences, bin, reverse_flag, logger):

        self.existing_models = OrderedDict(sorted(self.existing_models.items(), reverse=reverse_flag))
        # get the trained model of the previous bin and continue training with the new bin data:
        # the most recent model should be the last key in dict
        final_existing_bin = self.existing_models.keys()[-1]
        logger.info("Loading " + final_existing_bin + " for online training.")
        # unpickle and load model
        pickled_model = DATABASE.getGridFS().get(self.existing_models[final_existing_bin]).read()
        model = pickle.loads(pickled_model)

        logger.info("Training online for " + bin)

        if "bi" in self.ngram:
            bigram = Phrases(new_sentences, min_count=10, threshold=10, delimiter=b' ')
            bigram_phraser = Phraser(bigram)
            model.build_vocab(bigram_phraser[new_sentences], update=True)
            model.train(bigram_phraser[new_sentences])

        elif "tri" in self.ngram:
            bigram = Phrases(new_sentences, min_count=10, threshold=10, delimiter=b' ')
            bigram_phraser = Phraser(bigram)
            trigram = Phrases(bigram_phraser[new_sentences], min_count=10, threshold=10, delimiter=b' ')
            trigram_phraser = Phraser(trigram)
            model.build_vocab(trigram_phraser[bigram_phraser[new_sentences]], update=True)
            model.train(trigram_phraser[bigram_phraser[new_sentences]])
        else:
            model.build_vocab(new_sentences, update=True)
            # model.train(new_sentences, total_examples=model.corpus_count, epochs=model.iter) # for gensim 2.3
            model.train(new_sentences)


        return model


    def _evaluateWordAnalogy(self, timestamp, model, logger):
        # http://iamaziz.github.io/blog/2015/11/02/word2vec-with-nltk-retrain-and-evaluate/
        analogy_file_path = EngineConstants.EVAL_DATA_PATH + "analogy/questions-words.txt"
        evals = open(analogy_file_path, 'r').readlines()
        num_sections = len([l for l in evals if l.startswith(':')])
        num_sents = len(evals) - num_sections

        accuracy = model.wv.accuracy(analogy_file_path)

        sum_corr = len(accuracy[-1]['correct'])
        sum_incorr = len(accuracy[-1]['incorrect'])
        total = sum_corr + sum_incorr
        correct_perc = (sum_corr / total) * 100
        incorrect_perc = (sum_incorr / total) * 100

        if timestamp not in self.analogy_accuracies.keys():
            self.analogy_accuracies[timestamp] = {}
            self.analogy_accuracies[timestamp]["total"] = total
            self.analogy_accuracies[timestamp]["correct"] = sum_corr
            self.analogy_accuracies[timestamp]["incorrect"] = sum_incorr

        logger.info('Total: {} out of {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, num_sents, correct_perc, incorrect_perc))


    def _evaluateWordSimilarity(self, timestamp, model, logger):

        ws_file_dir = EngineConstants.EVAL_DATA_PATH + "ws/"
        method_paths = {'WS353'     : os.path.join(ws_file_dir, 'ws353.txt'),
                'MEN'               : os.path.join(ws_file_dir, 'bruni_men.txt'),
                'LUONG_RARE'        : os.path.join(ws_file_dir, 'luong_rare.txt'),
                'RADINSKY_MTURK'    : os.path.join(ws_file_dir, 'radinsky_mturk.txt'),
                'WS353_RELATEDNESS' : os.path.join(ws_file_dir, 'ws353_relatedness.txt'),
                'WS353_SIMILARITY'  : os.path.join(ws_file_dir, 'ws353_similarity.txt')
                }

        for method_tag, method_path in method_paths.iteritems():
            ms = ModelSimilarity()
            spearman_rho, used, total = ms.calculateWordSimilarity(model, method_path)
            if method_tag not in self.similarity_accuracies.keys():
                self.similarity_accuracies[method_tag] = {}
                self.similarity_accuracies[method_tag][timestamp] = {}
            elif timestamp not in self.similarity_accuracies[method_tag].keys():
                self.similarity_accuracies[method_tag][timestamp] = {}

            self.similarity_accuracies[method_tag][timestamp]["used"] = used
            self.similarity_accuracies[method_tag][timestamp]["total"] = total
            self.similarity_accuracies[method_tag][timestamp]["spearman_rho"] = spearman_rho

            if used is None or total is None or spearman_rho is None:
                logger.info('Spearman rho for %6s in %9s (%d/%d pairs): %f' % (method_tag, timestamp, used, total, spearman_rho))
            else:
                logger.info('Spearman rho for %6s in %9s (%s/%s pairs): %s' % (method_tag, timestamp, used, total, spearman_rho))


    def getAnalogyAccuracies(self):
        return self.analogy_accuracies

    def getWSAccuracies(self):
        return self.similarity_accuracies

    def getAverageAnalogyAccuracy(self):
        # returns first correct and second incorrect percentages
        vsm_total = 0
        vsm_corr = 0
        vsm_incorr = 0
        for timestamp in self.analogy_accuracies:
            vsm_total += self.analogy_accuracies[timestamp]["total"]
            vsm_corr += self.analogy_accuracies[timestamp]["correct"]
            vsm_incorr += self.analogy_accuracies[timestamp]["incorrect"]

        if not vsm_total:
            vsm_total = 1
        per_corr = (vsm_corr / vsm_total) * 100
        per_incorr = (vsm_incorr / vsm_total) * 100

        return per_corr, per_incorr

    def getAverageSimilarityAccuracyPerMethod(self, method):

        vsm_total = 0
        vsm_used = 0
        vsm_spearman = 0
        ts = 0
        for timestamp in self.similarity_accuracies[method]:
            curr_used = self.similarity_accuracies[method][timestamp]["used"]
            curr_tot = self.similarity_accuracies[method][timestamp]["total"]
            curr_spearman = self.similarity_accuracies[method][timestamp]["spearman_rho"]
            if curr_used != None and curr_tot != None and curr_spearman != None:
                vsm_used += curr_used
                vsm_total += curr_tot
                vsm_spearman += curr_spearman
                ts += 1

        if not vsm_total:
            vsm_total = 1
        per_used = (vsm_used / vsm_total) * 100
        if not ts:
            ts = 1
        average_rho = vsm_spearman / ts

        return vsm_used, vsm_total, per_used, average_rho