# Evolution of the tone of Artificial Intelligence Narratives

The aim of this project is to explore how the narratives regarding Artificial Intelligence (AI) have been changing throughout its history.
TILT API, which is developed to realise this project, is in fact a generic framework that combines the following approaches:
- Diachronic computational linguistics: Using context predicting distributional semantic models to mine word/topic associations and to create semantic vector spaces for each specified time interval
- Natural Language Processing: Using lexicon based unsupervised methods to analyse and track the changes in sentiment over time

The API currently has the following functionality:
- Creation and training of semantic vector spaces
- Comparison of various semantic vector spaces with regard to intrinsic evaluation (similarity and analogy tasks)
- POS tagging using Stanford CoreNLP Server
- Word-level sentiment analysis using 3 different lexicons (i.e. MPQA, SO-CAL, SentiWordNet 3.0)
- Visualisation of evolving semantic distance between words of interest through time
- Visualisation of semantic tracking using either PCA or t-SNE method
- Visualisation of evolving sentiment for a topic

## Getting Started

### Prerequisites

* Celery with RabbitMQ backend (https://tests4geeks.com/python-celery-rabbitmq-tutorial/)
* MongoDB (https://www.mongodb.com/download-center#community)
* NLTK (http://www.nltk.org/data.html)
* Gensim (https://radimrehurek.com/gensim/install.html)
* Stanford CoreNLP Server (https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)
+ Mailgun API (Optional - send notification email upon training completion)

See the [requirements.txt](requirements.txt) file for the library versions.

### Registration
In order to use the Mailgun API and receive emails, you will need to sign up and get your key from https://www.mailgun.com

### Configure

The following files need to be configured with your details or choices:

```
src/celery_tasks/celery_app.py
src/common/database.py
src/models/alerts/constants.py
```

### API Endpoint Reference

| Method | Endpoint | Usage | Returns | Example Request Body (JSON) |
| --- | --- |  --- |  --- | --- |
| GET | /api/experiments/<user_email> | Get experiments created using the *user_email* | User experiments (JSON) | - |
| GET | /api/experiments/public/ | Get experiments created publicly | Public experiments (JSON) | - |
| GET | /api/experiments/overview/<user_email> | Compare experiments created using the *user_email* | Components of Bokeh plot (HeatMap) to be embedded (JSON) | - |
| GET | /api/experiments/public/overview/ | Compare experiments created publicly | Components of Bokeh plot (HeatMap) to be embedded (JSON) | - |
| GET | /api/experiment/<experiment_id> | Get experiment having *experiment_id* | Experiment details (JSON) | - |
| GET | /api/experiment/run/<experiment_id> | Run experiment having *experiment_id* | Status information (Plain text) | - |
| GET | /api/experiment/delete/<experiment_id> | Delete experiment having *experiment_id* | Status information (Plain text) | - |
| POST | /api/experiment/run/ | Run experiment | Status information (Plain text) | ```{ "experimentId" : "a6a47948ef0a438bbb3db96ec9df7696"}``` |
| POST | /api/experiments/ | Get experiments created using the *user_email* | User experiments (JSON)  | ```{"userEmail" : "user@email.com"}``` |
| POST | /api/experiment/new/ | Configure and create experiment | Status information (Plain text) | ``` { "userEmail" : "user@email.com", "form" : [ { "experiment_display_title" : "Experiment title", "online_flag" : true, "training_algorithm" : "SGHS" } ] } ``` |
| POST | /api/experiment/visualise_semantic_distance/<experiment_id> | Visualise evolution of semantic distance between a keyword and aspects of interest | Components of Bokeh plot to be embedded (JSON) | ``` {"form" : [ { "keyword" : "artificial intelligence", "num_neighbours" : 10, "aspect_list[]" : ["military", "jobs", "transport", "efficiency"] } ] } ``` |
| POST | /api/experiment/visualise_semantic_tracking/<experiment_id> | Run experiment having *experiment_id* | Components of Bokeh plot to be embedded (JSON) | ``` { "form" : [ { "keyword" : "artificial intelligence", "num_neighbours" : 10, "visualisation_alg" : "pca" } ] } ``` |
| POST | /api/experiment/visualise_sentiment_analysis/<experiment_id> | Delete experiment having *experiment_id* | Components of Bokeh plot to be embedded (JSON) | ``` {"form" : [ { "keyword" : "artificial intelligence", "num_neighbours" : 10, "sentiment_lexicon" : "SO-CAL" } ] } ``` |

## Limitations

Existing sentiment lexicons consider only the most recent time. Therefore the calculation of sentiment of the historical words may not reflect the corresponding polarity.

## Authors

* **Aysenur Bilgin**

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Some of the code is adapted from the software used in Ruben Blom's BSc project (https://staff.fnwi.uva.nl/b.bredeweg/pdf/BSc/20152016/Blom.pdf)



