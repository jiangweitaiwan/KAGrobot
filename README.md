# KAGrobot

以知識圖譜（Neo4j）為核心的 AI 學習助手：結合 LangGraph 對話編排、語意檢索、Global / Personal Brain 視覺化，以及可互動的 Skill HTML Widget。

- **後端**：FastAPI + LangGraph + Gemini，部署於 Google Cloud Run  
- **前端**：Next.js + Three.js / D3，部署於 Vercel  
- **身分**：Firebase Auth；用量與會員額度存於 Firestore  

---

## 功能概覽

| 模組 | 說明 |
|------|------|
| 對話（Dialogue） | Socrates / Main Agent 雙模式，透過 orchestrator 編排工具呼叫 |
| Random Walk | 在知識圖譜上漫遊學習，與 dialogue 共用同一套 LangGraph checkpointer |
| Global Brain | 全域概念圖譜探索、BFS 展開、語意評分 |
| Personal Brain | 個人化知識權重與收藏 |
| Skills | GCS 載入技能包；支援沙箱執行（e2b）與互動式 HTML Widget |
| 會員 / 額度 | Guest / Normal / VIP 額度控管、`/me/usage`、後台管理 API |

---

## 專案結構

```
KAGrobot-e2b/
├── backend/                 # FastAPI 後端
│   ├── api/routes/          # REST / SSE 路由
│   ├── core/                # agents、LLM、skills、sessions、會員
│   ├── orchestrator/        # LangGraph 編排（dialogue + walk）
│   ├── graph_service/       # BFS、walk、layout
│   ├── retrieval/           # SBERT ONNX、候選檢索、路徑選擇
│   ├── tools/               # 知識、資料、skill code / widget
│   ├── app.py               # 應用工廠
│   ├── entry.py             # uvicorn 進入點
│   ├── settings.py          # 環境變數（pydantic-settings）
│   └── Dockerfile
├── frontend/                # Next.js 前端
│   ├── app/                 # 頁面（intro、dashboard、account…）
│   └── lib/                 # API client、Firebase、SSE
├── example_skill/           # Skill 範例（含 HTML widget）
├── capm-cost-of-equity-skill/
└── .github/workflows/       # Cloud Run 自動部署
```

更細的 Widget 落地說明見 [`README_HTML_WIDGET_CHANGES.md`](./README_HTML_WIDGET_CHANGES.md)；前端單獨說明見 [`frontend/README.md`](./frontend/README.md)。

---

## 技術棧

**Backend**  
Python 3.11 · FastAPI · LangGraph / LangChain · Gemini（Google AI 或 Vertex AI）· Neo4j · ONNX Runtime（SBERT）· Firestore · GCS · e2b · Firebase Admin

**Frontend**  
Next.js · React · TypeScript · Three.js · D3 · Firebase Auth

---

## 前置需求

- Python 3.11+
- Node.js 18+
- Neo4j 連線（Aura 或自架）
- Google API Key（或 Vertex AI 專案）
- Firebase 專案（前端 Auth；後端驗證 token）
- （選用）GCS buckets、e2b API key、LangSmith

---

## 本地開發

### 1. 後端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

在 `backend/` 建立 `.env`（變數名稱對應 `settings.py`），至少需要：

```env
GOOGLE_API_KEY=...
NEO4J_URI=neo4j+s://....databases.neo4j.io
NEO4J_PASSWORD=...
CORS_ALLOW_ORIGINS=http://localhost:3000

# HTML Widget（要啟用互動表單時必填）
WIDGET_TOKEN_SECRET=   # 例如: openssl rand -hex 32
FRONTEND_ORIGIN=http://localhost:3000
```

啟動：

```bash
python entry.py
# 預設 http://0.0.0.0:8000
```

Docker（與正式環境較接近，會從 GCS 拉取 SBERT 模型）：

```bash
cd backend
docker build -t kagrobot-backend .
docker run --env-file .env -p 8080:8080 kagrobot-backend
```

### 2. 前端

```bash
cd frontend
cp .env.example .env.local
```

編輯 `.env.local`：

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

並依 `lib/firebase.ts` 設定 Firebase 相關公開設定（若尚未寫入環境變數）。

```bash
npm install
npm run dev
# http://localhost:3000
```

---

## 主要環境變數

### 後端（`backend/.env`）

| 變數 | 說明 |
|------|------|
| `GOOGLE_API_KEY` | Gemini API key（非 Vertex 時） |
| `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_CLOUD_PROJECT` | 改走 Vertex AI |
| `MAIN_AGENT_MODEL` / `SOCRATES_MODEL` | 模型名稱 |
| `NEO4J_URI` / `NEO4J_PASSWORD` | 知識圖譜 |
| `SBERT_ONNX_MODEL_PATH` / `SBERT_TOKENIZER_DIR` | 本地語意編碼模型 |
| `SKILLS_BUCKET_NAME` | Skill 套件 GCS bucket |
| `E2B_API_KEY` | e2b 沙箱（skill code 執行） |
| `WIDGET_TOKEN_SECRET` | Widget HMAC 密鑰 |
| `FRONTEND_ORIGIN` | 前端 origin（widget `postMessage`） |
| `BACKEND_BASE` | 後端公開 URL（注入 widget） |
| `CORS_ALLOW_ORIGINS` | 逗號分隔允許來源 |
| `QUOTA_ENFORCE` | 是否強制額度擋請求 |
| `FIRESTORE_*` | Firestore 專案 / 狀態庫 |
| `LANGCHAIN_*` | LangSmith tracing（選用） |

完整欄位與預設值見 [`backend/settings.py`](./backend/settings.py)。

### 前端（`frontend/.env.local`）

| 變數 | 說明 |
|------|------|
| `NEXT_PUBLIC_API_BASE_URL` | 後端 base URL（本機或 Cloud Run） |

---

## API 模組（後端）

| Router | 用途 |
|--------|------|
| `chat` | 對話回合 |
| `walk` | Random Walk 開始 / 串流 |
| `graph` / `concept` | 圖譜與概念查詢 |
| `brain` / `user_brain` | Global / Personal Brain |
| `users` | 使用者與 session |
| `skill_widget` | `POST /skill/widget/dispatch`（即時試算） |
| `account` | `/me/usage` 等會員資訊 |
| `admin` | 後台管理 |

---

## 部署

### 後端 → Cloud Run

推送 `backend/**` 至 `main` / `KAGrobot/v2` / `KAGrobot/v3` 會觸發 [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml)：

1. OIDC 驗證 GCP  
2. Cloud Build 建置並推送映像  
3. 部署至 `asia-east1` 的 Cloud Run 服務 `kagrobot`  

部署前請在 Cloud Run 設定上述環境變數與服務帳號權限（GCS 簽名 URL 需 `roles/iam.serviceAccountTokenCreator` 等）。

### 前端 → Vercel

1. Import 此 repo，Root Directory 設為 `frontend`  
2. 設定 `NEXT_PUBLIC_API_BASE_URL` 為 Cloud Run URL  
3. Deploy  

後端 `CORS_ALLOW_ORIGINS` 需包含 Vercel 網域；正式環境請一併設定 `FRONTEND_ORIGIN`。

---

## Skill 套件

Skill 存放於 GCS（`SKILLS_BUCKET_NAME`）。本機可參考：

- [`example_skill/`](./example_skill/) — Black-Litterman 等示範（含 `html/manifest.yaml` 與 compute scripts）  
- [`capm-cost-of-equity-skill/`](./capm-cost-of-equity-skill/) — CAPM 權益成本範例  

互動 Widget 流程：`show_interactive_html` → 前端 iframe → `/skill/widget/dispatch` 即時試算 → 使用者提交後走下一輪 `/chat`。詳見 [`README_HTML_WIDGET_CHANGES.md`](./README_HTML_WIDGET_CHANGES.md)。

---

## 授權與注意事項

- 本專案含雲端金鑰與資料庫連線設定，請勿將 `.env`、服務帳號金鑰提交進 git。  
- 生產環境務必設定非空的 `WIDGET_TOKEN_SECRET` 與明確的 `FRONTEND_ORIGIN`。  
