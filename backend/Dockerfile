# --- 第一階段：模型搬運工 (model-fetcher) ---
  FROM google/cloud-sdk:slim AS model-fetcher
  WORKDIR /tmp
  
  # 請將這裡的路徑換成你實際的 GCS 路徑
  # 建議使用 gs://your-bucket-name/sbert_onnx.tar.gz
  ARG GCS_MODEL_URL="gs://kagrobot-models/sbert_onnx.tar.gz"
  
  # 從 GCS 下載並解壓縮
  # 注意：假設你的 tar.gz 裡面直接就是模型檔案，或是帶有 backend/sbert_onnx 路徑
  RUN gsutil cp ${GCS_MODEL_URL} . && \
      mkdir -p extracted_model && \
      tar -xzf sbert_onnx.tar.gz -C extracted_model
  
  # --- 第二階段：正式執行環境 (Runtime) ---
  FROM python:3.11-slim
  
  ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PIP_NO_CACHE_DIR=1
  
  WORKDIR /app
  
  # 安裝基礎系統套件
  RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
  
  # 1. 先從搬運工那裡把「模型」拿過來 (放在最前面，因為模型不常變動)
  # 注意：這裡的路徑需根據你 tar 壓縮時的結構調整
  COPY --from=model-fetcher /tmp/extracted_model /app/sbert_onnx
  
  # 2. 安裝 Python 依賴
  COPY requirements.txt /app/requirements.txt
  RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt
  
  # 3. 複製剩餘的後端程式碼
  COPY . /app
  
  EXPOSE 8080
  
  # 設定環境變數
  ENV HOST=0.0.0.0 \
      PORT=8080 \
      # 這裡對應你程式碼讀取模型的路徑
      SBERT_ONNX_MODEL_PATH=/app/sbert_onnx/sbert_multilingual.onnx \
      SBERT_TOKENIZER_DIR=/app/sbert_onnx
  
  # 啟動指令
  CMD ["sh", "-c", "uvicorn entry:api --host ${HOST} --port ${PORT}"]