import uuid
import src.models.configurations.constants as ConfigurationConstants
import src.models.configurations.errors as ConfigurationErrors
from src.common.database import Database

DATABASE = Database()

__author__ = 'abilgin'

class Configuration(object):

    def __init__(self, user_email, **kwargs):

        if 'form' in kwargs:
            # creation from the web
            self.user_email = user_email
            self._id = uuid.uuid4().hex
            self.render_form(kwargs['form'])
        elif 'configuration' in kwargs:
            # request from the creation of experiment
            self.__dict__.update(kwargs['configuration'].__dict__)
            self.user_email = user_email
        else:
            # default constructor from the database
            self.__dict__.update(kwargs)
            self.user_email = user_email


    def __eq__(self, other):

        if other is None:
            return False

        return self.corpus_list == other.corpus_list and self.genre == other.genre and self.bin_interval == other.bin_interval \
                and self.ngram == other.ngram and self.reverse_flag == other.reverse_flag and self.init_flag == other.init_flag \
                and self.until_flag == other.until_flag and self.online_flag == other.online_flag and self.align_flag == other.align_flag \
                and self.training_algorithm == other.training_algorithm and self.negative == other.negative and self.dimensions == other.dimensions \
                and self.window_size == other.window_size and self.iter == other.iter and self.sample == other.sample and self.min_count == other.min_count


    def render_form(self, form):

        # corpus pre-processing
        corpus_list_str = form['corpus_list'] if 'corpus_list' in form else ConfigurationConstants.DEFAULT_CORPUS_LIST
        self.corpus_list = corpus_list_str.replace(" ", "").split(",")
        self.genre = form['genre'] if 'genre' in form else ConfigurationConstants.DEFAULT_GENRE
        self.bin_interval = int(form['bin_interval']) if 'bin_interval' in form else ConfigurationConstants.DEFAULT_BIN_INTERVAL
        self.ngram = form['ngram'] if 'ngram' in form else ConfigurationConstants.DEFAULT_NGRAM

        # vector space generation
        self.reverse_flag = 'reverse_flag' in form
        self.init_flag = 'init_flag' in form
        self.until_flag = 'until_flag' in form
        self.online_flag = 'online_flag' in form
        self.align_flag = 'align_flag' in form

        # word embedding parameters
        self.training_algorithm = form['training_algorithm'] if 'training_algorithm' in form else ConfigurationConstants.DEFAULT_TRAINING_ALG
        self.negative = int(form['negative']) if 'negative' in form else ConfigurationConstants.DEFAULT_NEGATIVE
        self.dimensions = int(form['dimensions']) if 'dimensions' in form else ConfigurationConstants.DEFAULT_DIMENSIONS
        self.window_size = int(form['window_size']) if 'window_size' in form else ConfigurationConstants.DEFAULT_WINDOW_SIZE
        self.iter = int(form['iter']) if 'iter' in form else ConfigurationConstants.DEFAULT_ITER
        self.sample = float(form['sample']) if 'sample' in form else ConfigurationConstants.DEFAULT_SAMPLE
        self.min_count = int(form['min_count']) if 'min_count' in form else ConfigurationConstants.DEFAULT_MIN_COUNT

    @classmethod
    def get_by_user_email(cls, user_email):
        return [cls(**elem) for elem in DATABASE.find(ConfigurationConstants.COLLECTION, {"user_email": user_email})]

    @staticmethod
    def is_config_unique(new_config):
        user_config_list = Configuration.get_by_user_email(new_config.user_email)

        for config in user_config_list:
            if config == new_config:
                raise ConfigurationErrors.ConfigAlreadyExistsError("The configuration already exists.")

        return True

    def save_to_db(self):
        DATABASE.update(ConfigurationConstants.COLLECTION, {"_id": self._id}, self.__dict__)

    def delete(self):
        DATABASE.remove(ConfigurationConstants.COLLECTION, {"_id": self._id})

    @staticmethod
    def delete_by_experiment(experiment_dict):
        # get id of the configuration by subsethood
        user_config_list = Configuration.get_by_user_email(experiment_dict['user_email'])
        exp_list = experiment_dict.items()

        for config in user_config_list:
            conf_list = config.__dict__.items()
            if all(item in exp_list for item in conf_list if item[0] != "_id"):
                config.delete()
                break



