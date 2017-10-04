from __future__ import absolute_import
from src.celery_tasks.celery_app import celery_app
from celery.exceptions import Ignore

from src.models.experiments.experiment import Experiment

__author__ = 'abilgin'

@celery_app.task(bind=True)
def run_exp(self, exp_id):
    exp = Experiment.get_by_id(exp_id)
    if exp is None:
        self.update_state(state='FAILED', meta={'reason': 'Experiment does not exist!'})
        raise Ignore()

    if exp.run_started is not None and exp.run_finished is None:
        self.update_state(state='ALREADY_STARTED', meta={'reason': 'This experiment is already running...'})
        raise Ignore()
    elif exp.run_started is not None and exp.run_finished is not None:
        self.update_state(state='ALREADY_RUN', meta={'reason': 'This experiment has already completed running...'})
        raise Ignore()

    self.update_state(state='RUNNING', meta={'experiment_id': exp_id})
    exp.start_running()

@celery_app.task(bind=True)
def del_exp(self, exp_id):
    exp = Experiment.get_by_id(exp_id)
    if exp is None:
        self.update_state(state='FAILED', meta={'reason': 'Experiment does not exist!'})
        raise Ignore()

    exp.delete()

@celery_app.task(bind=True, trail=True)
def vis_asp_sem_dist(self, exp_id, keyword, num_neighbours, aspects):
    exp = Experiment.get_by_id(exp_id)
    self.update_state(state='VISUALISING', meta={'experiment_id': exp_id})
    return exp.visualise_aspect_based_semantic_distance(keyword, num_neighbours, aspects)

@celery_app.task(bind=True, trail=True)
def vis_sem_track(self, exp_id, keyword, num_neighbours, aspects, algorithm, tsne_perp, tsne_iter):
    exp = Experiment.get_by_id(exp_id)
    self.update_state(state='VISUALISING', meta={'experiment_id': exp_id})
    return exp.visualise_semantic_tracking(keyword, num_neighbours, aspects, algorithm, tsne_perp, tsne_iter)

@celery_app.task(bind=True, trail=True)
def vis_sentiment(self, exp_id, keyword, num_neighbours, lexicon, requested_corpus_list):
    exp = Experiment.get_by_id(exp_id)
    self.update_state(state='VISUALISING', meta={'experiment_id': exp_id})
    return exp.visualise_sentiment_analysis(keyword, num_neighbours,lexicon, requested_corpus_list)

