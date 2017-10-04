from src.common.database import Database
from src.engine import EngineConstants
from src.engine.naturallanguageprocessing.nlp import NLPOperator

DATABASE = Database()
""" This class is adapted from https://github.com/Noxeus/opinion-changes-thesis
"""
class LexiconBasedSentimentAnalyser:

    def __init__(self, experiment_id, keyword, num_neighbours):
        self.experiment_id = experiment_id
        self.keyword = keyword
        self.num_neighbours = num_neighbours

    def analyseUsingLexicon(self, lexicon, override=False):

        if not lexicon in [
            'SentiWordNet',
            'MPQA',
            'SO-CAL'
        ]:
            raise TypeError("Non supported lexicon.")

        if lexicon == "SentiWordNet":
            import src.engine.naturallanguageprocessing.lexiconBase.SentiWordNet as lexType
        elif lexicon == "MPQA":
            import src.engine.naturallanguageprocessing.lexiconBase.MPQA as lexType
        elif lexicon == "SO-CAL":
            import src.engine.naturallanguageprocessing.lexiconBase.SOCAL as lexType

        iterable = DATABASE.iter_collection(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'exp_id': self.experiment_id,
                                                                         'keyword': self.keyword,
                                                                         'num_neighbours': self.num_neighbours,
                                                                         'sentences': {'$exists': True}})

        for doc in iterable:
            if (not (lexicon in doc['sentiment'] and
                     lexicon in doc['sentence_polarities']) or
                    override):
                polarity, sentence_polarities = lexType.get_sentiment(doc)
                if polarity and sentence_polarities:
                    DATABASE.update(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'_id': doc['_id']}, {'$set': {
                             "sentiment." + lexicon: polarity,
                             'sentence_polarities.' + lexicon: sentence_polarities}
                    })

    def fromDBRunAnalysis(self):

        # word-level sentiment analysis
        # using all the selected sentences identify opinion words and aggregated polarity

        # initialize NLP operator
        nlp_op = NLPOperator()

        # loop over the documents for fully annotated dataset
        documents = DATABASE.iter_collection(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'exp_id': self.experiment_id,
                                                                          'keyword': self.keyword, 'num_neighbours': self.num_neighbours})
        for entry in documents:
            nlp_op.process(entry)

        self.analyseUsingLexicon("SO-CAL")
        self.analyseUsingLexicon("MPQA")
        self.analyseUsingLexicon("SentiWordNet")
