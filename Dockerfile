# Auto Insurance Assistant - production image (CPU only; the LLM is the Gemini API).
#   docker build -t insurance-assistant:latest .
#   docker compose up        (see docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # caches/config the libraries write at runtime, kept inside /app or /tmp
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers \
    HF_HOME=/app/models/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_AGENT_HINT=1

# OpenCV (pulled in by ultralytics) needs libGL/glib; python-magic (unstructured) needs libmagic
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) dependencies first so code changes don't reinstall them. CPU-only PyTorch keeps the image a few GB smaller.
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 2) the embedding model, downloaded at build time so pods start without internet access.
#    sentence-transformers 2.2.2 fetches the whole model repo (ONNX, OpenVINO, a duplicate .bin): ~760 MB.
#    Only model.safetensors is used, so the rest is deleted in the same layer (~90 MB left).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && cd "$SENTENCE_TRANSFORMERS_HOME"/sentence-transformers_all-MiniLM-L6-v2 \
    && rm -rf onnx openvino pytorch_model.bin tf_model.h5 rust_model.ot flax_model.msgpack \
    && python -c "from sentence_transformers import SentenceTransformer; print(SentenceTransformer('all-MiniLM-L6-v2').encode(['ok']).shape)"

# 3) application code + baked-in assets (YOLO weights, sample policies; see .dockerignore for what's excluded)
COPY insuranceasst/ /app/insuranceasst/
WORKDIR /app/insuranceasst

# 4) build-time setup: search index, hashed static files, Django's own tables (sessions/admin)
RUN python manage.py build_index \
    && DJANGO_DEBUG=0 DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput \
    && python manage.py migrate --noinput \
    && mkdir -p data/images/claims \
    && useradd --uid 10001 --create-home app \
    && chown -R app:app /app/insuranceasst/data /app/insuranceasst/db.sqlite3 /app/insuranceasst/faiss_store

USER app
EXPOSE 8000

# production defaults; override with -e / Kubernetes env. DJANGO_SECRET_KEY and LLM_API_KEY must be supplied.
ENV DJANGO_DEBUG=0 \
    DEMO_READONLY=1 \
    MLFLOW_TRACING=0 \
    LLM_BACKEND=openai \
    LLM_MODEL=gemini-3.5-flash-lite

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "insuranceasst.wsgi", "-c", "gunicorn.conf.py"]
