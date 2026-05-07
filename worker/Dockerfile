# Worker uses the exact same image as API
# (Celery tasks need all the same dependencies)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .

RUN mkdir -p /app/data/chroma_data /app/logs

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info"]