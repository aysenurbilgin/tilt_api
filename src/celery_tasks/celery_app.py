from __future__ import absolute_import
from celery import Celery

celery_app = Celery('src.celery_tasks',
             broker='amqp://user:pass@localhost/vhost',
             backend='rpc://',
             include=['src.celery_tasks.tasks'])