"""
Celery configuration for cartoon_project
"""

import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')

app = Celery('cartoon_project')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Optional configuration for production
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone=settings.TIME_ZONE,
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
