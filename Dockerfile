FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (kept minimal; uvicorn[standard] needs build tools only in edge cases)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8080

# Default runtime config; override at `docker run -e ...`
ENV HOST=0.0.0.0 \
    PORT=8080 \
    # Model files are copied into /app/sbert_onnx by default
    SBERT_ONNX_MODEL_PATH=/app/sbert_onnx/sbert_multilingual.onnx \
    SBERT_TOKENIZER_DIR=/app/sbert_onnx

CMD ["sh", "-c", "uvicorn entry:api --host ${HOST} --port ${PORT}"]

