"""
Celery configuration for async job processing
"""
from celery import Celery
import os

# Get Redis URL from environment
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Debug: Print what we're using (will show in Railway logs)
print(f"[CELERY CONFIG] Using Redis URL: {redis_url[:20]}..." if len(redis_url) > 20 else f"[CELERY CONFIG] Using Redis URL: {redis_url}")
print(f"[CELERY CONFIG] REDIS_URL env var is: {'SET' if 'REDIS_URL' in os.environ else 'NOT SET'}")

# Create Celery app
celery_app = Celery(
    'diamond_classifier',
    broker=redis_url,
    backend=redis_url,
    include=['tasks']  # Import tasks module
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to free memory
)

if __name__ == '__main__':
    celery_app.start()
