import gensim
import numpy as np
from scipy.stats.stats import spearmanr

# global parameters for word2vec
ALPHA = 0.01                   # initial learning rate, drops to min_alpha
MIN_ALPHA = 0.0001
CBOW_MEAN = 1                   # http://stackoverflow.com/questions/34249586/the-accuracy-test-of-word2vec-in-gensim


class ModelAlignment:

    def __init__(self):
        pass

    def smart_procrustes_align_gensim(self, base_embed, other_embed, words=None):
        """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
        Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
            (With help from William. Thank you!)
        First, intersect the vocabularies (see `intersection_align_gensim` documentation).
        Then do the alignment on the other_embed model.
        Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
        Return other_embed.
        If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
        """

        # make sure vocabulary and indices are aligned
        # in_base_embed, in_other_embed = ModelAlignment.intersection_align_gensim(base_embed, other_embed, words=words)
        in_base_embed, in_other_embed = self.intersection_align_gensim(base_embed, other_embed, words=words)

        # get the embedding matrices
        base_vecs = in_base_embed.wv.syn0norm
        other_vecs = in_other_embed.wv.syn0norm

        # just a matrix dot product with numpy
        m = other_vecs.T.dot(base_vecs)

        # SVD method from numpy
        u, _, v = np.linalg.svd(m)
        # another matrix operation
        ortho = u.dot(v)
        # Replace original array with modified one
        # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
        other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)

        return other_embed

    def intersection_align_gensim(self, m1, m2, words=None):
        """
        Intersect two gensim word2vec models, m1 and m2.
        Only the shared vocabulary between them is kept.
        If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
        Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
        These indices correspond to the new syn0 and syn0norm objects in both gensim models:
            -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
            -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
        The .vocab dictionary is also updated for each model, preserving the count but updating the index.
        """

        # Get the vocab for each model
        vocab_m1 = set(m1.wv.vocab.keys())
        vocab_m2 = set(m2.wv.vocab.keys())

        # Find the common vocabulary
        common_vocab = vocab_m1&vocab_m2
        if words: common_vocab&=set(words)

        # If no alignment necessary because vocab is identical...
        if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
            return (m1,m2)

        # Otherwise sort by frequency (summed for both)
        common_vocab = list(common_vocab)
        common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

        # Then for each model...
        for m in [m1,m2]:
            # Replace old syn0norm array with new one (with common vocab)
            indices = [m.wv.vocab[w].index for w in common_vocab]

            old_arr = m.wv.syn0norm
            if old_arr is None:
                old_arr = m.wv.syn0

            new_arr = np.array([old_arr[index] for index in indices])
            m.wv.syn0norm = m.wv.syn0 = new_arr

            # Replace old vocab dictionary with new one (with common vocab)
            # and old index2word with new one
            m.wv.index2word = common_vocab
            old_vocab = m.wv.vocab

            new_vocab = {}
            for new_index,word in enumerate(common_vocab):
                old_vocab_obj=old_vocab[word]
                new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
            m.wv.vocab = new_vocab

        return (m1,m2)


class ModelSimilarity:

    def __init__(self):
        pass

    def calculateWordSimilarity(self, model, method_path):

        # Return Spearman rho for a given key -> vector embedding mapping
        # against data in standard similarity test file in method_path;
        # used and total pairs with user supplied scores from the file are
        # also returned
        # https://xdata-skylark.github.io/libskylark/examples/randsvd_embeddings.html

        test_data = []
        with open(method_path) as f:
            for line in f:
                x, y, sim = line.strip().lower().split()
                test_data.append(((x, y), sim))
        results = []
        misses = 0
        for i, ((x, y), sim) in enumerate(test_data):
            try:
                results.append((self.getSimilarityForWords(x, y, model), sim))
            except:
                misses += 1
                pass

        res = zip(*results)
        if len(res) != 2:
            return None, None, None
        actual, expected = zip(*results)
        total = i + 1
        used  = total - misses
        return spearmanr(actual, expected)[0], used, total

    def getSimilarityForWords(self, word1, word2, model):
        # For a given mapping word->vector word_vectors return
        # the similarity of word1 and word2
        # https://xdata-skylark.github.io/libskylark/examples/randsvd_embeddings.html

        v1 = model.wv[word1]
        v2 = model.wv[word2]
        return np.dot(v1, v2)