"""Factory for ChatGoogleGenerativeAI (Vertex AI or Gemini Developer API).

【Phase 0】所有 LLM 都從這裡建立，因此這裡是 token 計量的唯一攔截點：
每個實例掛一個 UsageCallbackHandler(feature)，feature 用來在用量報表中
區分「哪個功能燒掉的」：socratic / walk / main_agent / suggest / other。
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from core.usage import UsageCallbackHandler
from settings import Settings, get_settings


def create_chat_google_generative_ai(
    *,
    model: str,
    temperature: float = 0.0,
    api_key: str = "",
    settings: Settings | None = None,
    feature: str = "other",
) -> ChatGoogleGenerativeAI:
    """Build ChatGoogleGenerativeAI from settings.

    Vertex AI when ``google_genai_use_vertexai`` is true and ``google_cloud_project``
    is set; otherwise Gemini Developer API with ``google_api_key``.

    feature : 用量歸屬的功能名稱（不可含 "." 或 "/"，會成為 Firestore 欄位名）。
    """
    s = settings or get_settings()
    key = api_key or s.google_api_key
    callbacks = [UsageCallbackHandler(feature)]

    if s.google_genai_use_vertexai and s.google_cloud_project:
        kwargs: dict = {
            "model": model,
            "temperature": temperature,
            "vertexai": True,
            "project": s.google_cloud_project,
            "location": s.google_cloud_location,
            "callbacks": callbacks,
        }
        if key:
            kwargs["google_api_key"] = key
        return ChatGoogleGenerativeAI(**kwargs)

    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is required when Vertex AI is disabled "
            "(GOOGLE_GENAI_USE_VERTEXAI=false or GOOGLE_CLOUD_PROJECT unset)."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temperature,
        vertexai=False,
        callbacks=callbacks,
    )
