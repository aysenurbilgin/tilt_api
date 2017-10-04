import datetime
import requests
from src.common.database import Database
import src.models.experiments.constants as ExperimentConstants
import src.models.alerts.constants as AlertConstants
import src.engine.EngineConstants as EngineConstants
from src.engine.distributionalsemantics.word2vecVSModel import word2vecVSModel
from src.engine.visualisation.ModelVisualiser import ModelVisualiser
from src.models.configurations.configuration import Configuration

DATABASE = Database()

__author__ = 'abilgin'

class Experiment(word2vecVSModel):

    def __init__(self, user_email, display_title, public_flag, **kwargs):

        if 'configuration' in kwargs:
            # creation from the web
            word2vecVSModel.__init__(self, user_email=user_email, **kwargs)
            self.created = datetime.datetime.utcnow()
            self.run_started = None
            self.run_finished = None
        else:
            # default constructor from the database
            self.__dict__.update(kwargs)

        self.user_email = user_email
        self.display_title = display_title
        self.public_flag = public_flag

    def __repr__(self):
        return "<Experiment {}>".format(self.display_title)

    def save_to_db(self):
        DATABASE.update(ExperimentConstants.COLLECTION, {"_id": self._id}, self.__dict__)

    @classmethod
    def get_by_id(cls, id):
        exp = DATABASE.find_one(ExperimentConstants.COLLECTION, {"_id": id})
        return cls(**exp) if exp is not None else exp

    @classmethod
    def get_by_title(cls, title):
        exp = DATABASE.find_one(ExperimentConstants.COLLECTION, {"display_title": title})
        return cls(**exp) if exp is not None else exp

    @classmethod
    def get_by_user_email(cls, user_email):
        exp_list = DATABASE.find(ExperimentConstants.COLLECTION, {"user_email": user_email})
        if exp_list is not None:
            return [cls(**elem) for elem in exp_list]
        else:
            return None

    def delete(self):
        id = self._id
        dictionary = self.__dict__
        existing_models = self.existing_models
        DATABASE.remove(ExperimentConstants.COLLECTION, {'_id': id})
        Configuration.delete_by_experiment(dictionary)
        for file_id in existing_models.values():
            DATABASE.getGridFS().delete(file_id)
        documents = DATABASE.iter_collection(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'exp_id': id})
        for entry in documents:
            DATABASE.remove(EngineConstants.SELECTED_SENTENCES_COLLECTION, {'_id': entry['_id']})

    @classmethod
    def get_public_experiments(cls):
        return [cls(**elem) for elem in
                DATABASE.find(ExperimentConstants.COLLECTION, {"public_flag": True, "run_finished": {"$ne" : None}})]

    @classmethod
    def get_finished_experiments(cls):
        return [cls(**elem) for elem in
                DATABASE.find(ExperimentConstants.COLLECTION, {"": True, "run_finished": {"$ne": None}})]

    def start_running(self):
        # update the timestamp
        self.run_started = datetime.datetime.utcnow()
        self.save_to_db()

        # call the training method
        if self.align_flag:
            self.createAlignedVSModel(self.until_flag, self.reverse_flag, self.online_flag)
        else:
            self.createVSModel(self.until_flag, self.reverse_flag, self.online_flag)

        self.run_finished = datetime.datetime.utcnow()
        self.save_to_db()
        # send an email to the user upon training completion
        self.send_email()

    def send_email(self):
        return requests.post(
            AlertConstants.URL,
            auth=("api", AlertConstants.API_KEY),
            data={
                "from": AlertConstants.FROM,
                "to": AlertConstants.TO, #self.user_email
                "subject": "Training completed for {}!".format(self.display_title),
                "text": "Start your Travel In Linguistics Time ({})...".format("http://host:port/experiments/{}".format(self._id))
            }
        )

    def visualise_aspect_based_semantic_distance(self, keyword, num_neighbours, aspects):
        mv = ModelVisualiser(self, keyword, num_neighbours)
        return mv.getKeywordErrorStatus(), mv.distanceBasedAspectVisualisation(aspects)

    def visualise_semantic_tracking(self, keyword, num_neighbours, aspects, algorithm, tsne_perp, tsne_iter):
        mv = ModelVisualiser(self, keyword, num_neighbours)
        return mv.getKeywordErrorStatus(), mv.timeTrackingVisualisation(aspects, algorithm, tsne_perp, tsne_iter)

    def visualise_sentiment_analysis(self, keyword, num_neighbours, lexicon, requested_corpus_list):
        mv = ModelVisualiser(self, keyword, num_neighbours)
        if mv.getKeywordErrorStatus() == EngineConstants.ALL_KEYERR:
            return mv.getKeywordErrorStatus(), None, None
        return mv.getKeywordErrorStatus(), mv.sentimentVisualisation(lexicon, requested_corpus_list)
