FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (avoids 3GB CUDA download)
RUN pip install --no-cache-dir --timeout=120 \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install everything else with longer timeout
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chroma_data /app/logs /app/eval_results

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]