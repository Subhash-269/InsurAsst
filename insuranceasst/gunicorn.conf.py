# Gunicorn settings for the container (Linux). `gunicorn insuranceasst.wsgi -c gunicorn.conf.py`
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# One process: the embedding model, FAISS index and YOLO (~1-2 GB) load once and are shared by threads.
# Threads keep the server responsive while answers stream. Scale out with more pods, not more workers.
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "8"))

# streamed answers and the first (model-loading) request can take a while
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"   # stdout, collected by Docker / Kubernetes
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
