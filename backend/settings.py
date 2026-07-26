from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime / server ---
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_allow_origins: str = Field(default="*", alias="CORS_ALLOW_ORIGINS")

    # --- LLM ---
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    main_agent_model: str = Field(default="gemini-2.5-flash", alias="MAIN_AGENT_MODEL")
    socrates_model: str = Field(default="gemini-2.5-flash", alias="SOCRATES_MODEL")
    # Vertex AI（設 GOOGLE_GENAI_USE_VERTEXAI=true 且 GOOGLE_CLOUD_PROJECT 時啟用）
    google_genai_use_vertexai: bool = Field(
        default=False, alias="GOOGLE_GENAI_USE_VERTEXAI"
    )
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(
        default="us-central1", alias="GOOGLE_CLOUD_LOCATION"
    )

    # --- Neo4j ---
    neo4j_uri: str = Field(default="neo4j+s://124606a6.databases.neo4j.io", alias="NEO4J_URI")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    # --- Local SBERT ONNX ---
    sbert_onnx_model_path: str = Field(
        default="",
        alias="SBERT_ONNX_MODEL_PATH",
        description="ONNX model path (e.g. /app/sbert_onnx/sbert_multilingual.onnx)",
    )
    sbert_tokenizer_dir: str = Field(
        default="",
        alias="SBERT_TOKENIZER_DIR",
        description="Tokenizer directory (e.g. /app/sbert_onnx)",
    )

    # --- LangSmith / LangChain tracing ---
    langchain_tracing_v2: Optional[str] = Field(default="true", alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: Optional[str] = Field(default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(default="main", alias="LANGCHAIN_PROJECT")

    # --- Optional external data ---
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    bea_api_key: str = Field(
        default="",
        alias="BEA_API_KEY",
        description="BEA（Bureau of Economic Analysis）API key，供 tools/connectors/bea.py 使用",
    )

    # --- Python sandbox (GCP Cloud Function) ---
    python_sandbox_url: str = Field(
        default="",
        alias="PYTHON_SANDBOX_URL",
        description=(
            "GCP Cloud Function URL for remote Python execution "
            "(e.g. https://<region>-<project>.cloudfunctions.net/python_sandbox_worker)"
        ),
    )

    # --- GCS skills ---
    skills_bucket_name: str = Field(
        default="kagrobot-skills",
        alias="SKILLS_BUCKET_NAME",
    )

    # --- GCS results_dict (GlobalBrain layout) ---
    gcs_bucketforresult: str = Field(
        default="kagrobot-result",
        alias="GCS_BUCKETFORRESULT",
        description="GCS bucket that stores results_dict object at bucket root (object name: results_dict)",
    )

    results_pkl_path: str = Field(
        default="results_dict.pkl",
        alias="RESULTS_PKL_PATH",
        description="Path to results_dict.pkl for GlobalBrain layout",
    )    

    skill_outputs_bucket_name: str = Field(
        default = "skill_outputs_bucket",
        description = "GCS bucket for sandbox output"

    )

    def resolved_sbert_paths(self, *, base_dir: Optional[Path] = None) -> tuple[str, str]:
        """Return (model_path, tokenizer_dir) with sane defaults."""
        bd = base_dir or Path(".").resolve()
        model_path = self.sbert_onnx_model_path or str(bd / "sbert_onnx" / "sbert_multilingual.onnx")
        tokenizer_dir = self.sbert_tokenizer_dir or str(bd / "sbert_onnx")
        return model_path, tokenizer_dir

    # --- e2b sandbox (per-skill 隔離執行環境) ---
    e2b_api_key: str = Field(default="", alias="E2B_API_KEY")
    e2b_skill_template: str = Field(
        default="code-interpreter-v1", alias="E2B_SKILL_TEMPLATE"
    )

    rc_ranking_pool_size: int = Field(
        default=15, alias="RC_RANKING_POOL_SIZE",
        description="Ranking pool 的 top-K 大小，供前端 batch expand 使用",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()