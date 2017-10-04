import json

from flask_cors import CORS
from flask import (
    Flask,
    request,
    jsonify,
    abort
)
from src.common.database import Database
from src.celery_tasks.tasks import run_exp, del_exp, vis_asp_sem_dist, vis_sem_track, vis_sentiment
from src.models.configurations.configuration import Configuration
import src.models.configurations.errors as ConfigurationErrors
from src.models.experiments.experiment import Experiment
from src.engine.comparison.ExperimentComparator import ExperimentComparator
import time
from bokeh.resources import INLINE

import src.engine.EngineConstants as EC

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'topsecret!'
CORS(app, resources={r"/api/*": {"origins": "*"}})

__author__ = 'abilgin'

@app.route('/api/experiments/<string:user_email>', methods=['GET'])
def get_experiments(user_email):
    db = Database()
    experiments = list(db.find("experiments", {"user_email": user_email}, {"existing_models": 0}))
    # 200 OK
    return jsonify(experiments), 200

@app.route('/api/experiments/', methods=['POST'])
def get_experiments_by_post():
    if not request.json or not 'userEmail' in request.json:
        abort(400)

    user_email = request.json['userEmail']
    return get_experiments(user_email)

@app.route('/api/experiments/public/', methods=['GET'])
def get_public_experiments():
    db = Database()
    experiments = list(db.find("experiments", {"public_flag": True}, {"_id": 0, "existing_models": 0}))
    # 200 OK
    return jsonify(experiments), 200

@app.route('/api/experiment/new/', methods=['POST'])
def create_experiment():
    if not request.json or not 'userEmail' in request.json or not 'form' in request.json:
        abort(400)

    if request.json['userEmail'] == "":
        abort(400)

    user_email = request.json['userEmail']
    form = request.json['form'][0]
    configuration = Configuration(user_email=user_email, form=form)
    try:
        if Configuration.is_config_unique(configuration):
            configuration.save_to_db()
            display_title = form['experiment_display_title']
            public_flag = 'public_flag' in form
            experiment = Experiment(user_email=user_email, display_title=display_title, public_flag=public_flag,
                                    **dict(configuration=configuration))
            experiment.save_to_db()
    except ConfigurationErrors.ConfigAlreadyExistsError as e:
            # 412 Precondition Failed
            return e.message, 412

    # 201 Created
    return "Experiment created", 201

@app.route('/api/experiment/<string:experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    db = Database()
    experiment = db.find_one("experiments", {"_id": experiment_id}, {"_id": 0, "existing_models": 0})
    # 200 OK
    return jsonify(experiment), 200

@app.route('/api/experiment/run/<string:experiment_id>', methods=['GET'])
def run_experiment(experiment_id):
    task = run_exp.delay(experiment_id)
    time.sleep(0.5)

    if task.state == 'PENDING':
        # 202 Accepted
        return 'Experiment is queued.', 202
    elif task.state == 'RUNNING':
        # 200 OK
        return "Experiment started running", 200

    # 412 Precondition Failed
    return task.result['reason'], 412

@app.route('/api/experiment/run/', methods=['POST'])
def run_experiment_by_post():
    if not request.json or not 'experimentId' in request.json:
        # 400 Bad Request
        abort(400)

    experiment_id = request.json['experimentId']
    return run_experiment(experiment_id)

@app.route('/api/experiment/visualise_semantic_distance/<string:experiment_id>', methods=['GET','POST'])
def visualise_semantic_distance(experiment_id):
    if not request.json or not 'form' in request.json:
        abort(400)

    form = request.json['form'][0]
    keyword = form['keyword']
    num_neighbours = form['num_neighbours']
    aspects = form['aspect_list[]']
    # take unique aspect words without empty strings
    aspects = filter(bool, set(aspects))

    task = vis_asp_sem_dist.delay(experiment_id, keyword, num_neighbours, aspects)
    task.wait()

    warning_message = ""
    if len(task.result) > 1:
        keyword_error = task.result[0]
        if keyword_error == EC.ALL_KEYERR:
            # 412 Precondition Failed
            return 'Keyword "' + keyword + '" could not be found in the semantic space...', 412
        else:
            if keyword_error == EC.PARTIAL_KEYERR:
                # 200 OK
                warning_message = 'Keyword "' + keyword + '" could not be found in some subsets of the semantic space...'

            script = task.result[1][0]
            div = task.result[1][1]
            words_not_found = task.result[1][2]
            for word in words_not_found:
                warning_message = 'Aspect "' + word + '" could not be found in the semantic space...'

            result = {'plot_script':script, 'plot_div':div, 'js_resources':INLINE.render_js(),
                       'css_resources':INLINE.render_css(), 'warning' : warning_message,
                       'mimetype':'text/html'}
            # 200 OK
            return json.dumps(result), 200

@app.route('/api/experiment/visualise_semantic_tracking/<string:experiment_id>', methods=['GET','POST'])
def visualise_semantic_tracking(experiment_id):
    if not request.json or not 'form' in request.json:
        abort(400)

    form = request.json['form'][0]
    keyword = form['keyword']
    num_neighbours = form['num_neighbours']
    aspects = form['aspect_list[]'] if 'aspect_list[]' in form else []
    # take unique aspect words without empty strings
    aspects = filter(bool, set(aspects))
    algorithm = form['visualisation_alg']
    tsne_perp = int(form['perplexity']) if 'perplexity' in form else ""
    tsne_iter = int(form['iterations']) if 'iterations' in form else ""

    task = vis_sem_track.delay(experiment_id, keyword, num_neighbours, aspects, algorithm, tsne_perp, tsne_iter)
    task.wait()

    warning_message = ""
    if len(task.result) > 1:
        keyword_error = task.result[0]
        if keyword_error == EC.ALL_KEYERR:
            # 412 Precondition Failed
            return 'Keyword "' + keyword + '" could not be found in the semantic space...', 412
        else:
            if keyword_error == EC.PARTIAL_KEYERR:
                # 200 OK
                warning_message = 'Keyword "' + keyword + '" could not be found in some subsets of the semantic space...'

            script = task.result[1][0]
            div = task.result[1][1]

            result = {'plot_script':script, 'plot_div':div, 'js_resources':INLINE.render_js(),
                       'css_resources':INLINE.render_css(), 'warning' : warning_message,
                       'mimetype':'text/html'}
            # 200 OK
            return json.dumps(result), 200

@app.route('/api/experiment/visualise_sentiment_analysis/<string:experiment_id>', methods=['GET','POST'])
def visualise_sentiment_analysis(experiment_id):
    if not request.json or not 'form' in request.json:
        abort(400)

    form = request.json['form'][0]
    keyword = form['keyword']
    num_neighbours = form['num_neighbours']
    lexicon = form['sentiment_lexicon']
    if 'requested_corpora' not in form:
        requested_corpora = []
    else:
        requested_corpora = form.getlist('requested_corpora')

    task = vis_sentiment.delay(experiment_id, keyword, num_neighbours, lexicon, requested_corpora)
    task.wait()

    warning_message = ""
    if len(task.result) > 1:
        keyword_error = task.result[0]
        if keyword_error == EC.ALL_KEYERR:
            # 412 Precondition Failed
            return 'Keyword "' + keyword + '" could not be found in the semantic space...', 412
        else:
            if keyword_error == EC.PARTIAL_KEYERR:
                # 200 OK
                warning_message = 'Keyword "' + keyword + '" could not be found in some subsets of the semantic space...'

            script = task.result[1][0]
            div = task.result[1][1]

            result = {'plot_script':script, 'plot_div':div, 'js_resources':INLINE.render_js(),
                       'css_resources':INLINE.render_css(), 'warning' : warning_message,
                       'mimetype':'text/html'}
            # 200 OK
            return json.dumps(result), 200

@app.route('/api/experiments/public/overview/', methods=['GET'])
def public_overview():
    # call overview method with the public experiments
    experiments = Experiment.get_public_experiments()
    if len(experiments) == 0:
        return 'There are no public experiments... ', 400
    return overview(experiments)

@app.route('/api/experiments/overview/<string:user_email>',  methods=['GET'])
def user_experiments_overview(user_email):
    # call overview method with the experiments that belong to the user
    experiments = Experiment.get_by_user_email(user_email)
    if len(experiments) == 0:
        return 'There are no experiments for this user or the user does not exist... ', 400
    return overview(experiments)

@app.route('/api/experiment/delete/<string:experiment_id>', methods=['GET'])
def delete_experiment(experiment_id):
    task = del_exp.delay(experiment_id)
    time.sleep(1)

    exp = Experiment.get_by_id(experiment_id)
    if exp is None:
        # 200 OK
        return "Experiment deleted", 200

def overview(experiments):
    if len(experiments) > 1:
        comparator = ExperimentComparator(experiments)
        script, div = comparator.evalComparison()

        result = {'plot_script': script, 'plot_div': div, 'js_resources': INLINE.render_js(),
                  'css_resources': INLINE.render_css(), 'mimetype': 'text/html'}
        # 200 OK
        return json.dumps(result), 200

    # 412 Precondition Failed
    return 'There are not enough experiments to compare...', 412