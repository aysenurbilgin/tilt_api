import os
import errno
from os.path import join
from src.__init__ import ROOT
import logging

__author__ = 'abilgin'

ROOT_DATA = '/path/to/corpora/'

APP_LOG_PATH = join(ROOT, '../logs/app.log')
EXP_LOG_PATH = join(ROOT, '../logs/experiments/')

# Word2Vec PATHs
TRAINING_PATH = join(ROOT_DATA, 'training/')
EVAL_DATA_PATH = join(ROOT_DATA, 'eval/')

# Sentiment analysis source path
ORIGINAL_SOURCE_PATH = join(ROOT_DATA, 'original/')

NO_KEYERR = 0
PARTIAL_KEYERR = 1
ALL_KEYERR = 2
NO_DATA = 3

SELECTED_SENTENCES_COLLECTION = "selected_sentences"

class EngineUtils(object):

    def __init__(self):
        pass

    def setup_logger(self, name, log_file, level=logging.INFO):

        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

        if not os.path.exists(os.path.abspath(log_file)):
            try:
                os.makedirs(os.path.dirname(log_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger
