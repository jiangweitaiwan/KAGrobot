#!/bin/bash

# --- 1. 設定變數 (請根據你的實際資訊修改) ---
PROJECT_ID="vivid-lane-477209-u3"
REGION="asia-east1"
REPO_NAME="my-repo"
IMAGE_NAME="kagrobot"
SERVICE_NAME="kagrobot"

# 完整的映像檔路徑
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "🚀 開始部署流程：${SERVICE_NAME}..."

# --- 2. 構建 Docker 映像檔 ---
# 注意：我們在根目錄執行，但 Dockerfile 在 backend 資料夾內
echo "📦 正在構建 Docker 映像檔..."
docker build -t $IMAGE_URL -f backend/Dockerfile ./backend

if [ $? -ne 0 ]; then
    echo "❌ Docker 構建失敗，請檢查代碼或 Dockerfile。"
    exit 1
fi

# --- 3. 推送到 Artifact Registry ---
echo "⬆️ 正在將映像檔推送到 Google Cloud..."
docker push $IMAGE_URL

if [ $? -ne 0 ]; then
    echo "❌ 推送失敗，請確認是否已執行 gcloud auth configure-docker ${REGION}-docker.pkg.dev"
    exit 1
fi

# --- 4. 部署到 Cloud Run ---
# 這裡使用繼承模式，不重複指定 Secrets，除非你有更動
echo "🌐 正在更新 Cloud Run 服務..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_URL \
  --region $REGION \
  --platform managed

if [ $? -eq 0 ]; then
    echo "✅ 部署成功！"
    # 自動取得並顯示服務網址
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')
    echo "🔗 後端網址: ${SERVICE_URL}"
else
    echo "❌ Cloud Run 部署失敗。"
    exit 1
fi