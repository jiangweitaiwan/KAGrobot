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
    langchain_tracing_v2: Optional[str] = Field(default=None, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: Optional[str] = Field(default=None, alias="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(default=None, alias="LANGCHAIN_PROJECT")

    # --- Optional external data ---
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    def resolved_sbert_paths(self, *, base_dir: Optional[Path] = None) -> tuple[str, str]:
        """Return (model_path, tokenizer_dir) with sane defaults."""
        bd = base_dir or Path(".").resolve()
        model_path = self.sbert_onnx_model_path or str(bd / "sbert_onnx" / "sbert_multilingual.onnx")
        tokenizer_dir = self.sbert_tokenizer_dir or str(bd / "sbert_onnx")
        return model_path, tokenizer_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
