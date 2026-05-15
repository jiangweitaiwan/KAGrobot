"""Factory for ChatGoogleGenerativeAI (Vertex AI or Gemini Developer API)."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from settings import Settings, get_settings


def create_chat_google_generative_ai(
    *,
    model: str,
    temperature: float = 0.0,
    api_key: str = "",
    settings: Settings | None = None,
) -> ChatGoogleGenerativeAI:
    """Build ChatGoogleGenerativeAI from settings.

    Vertex AI when ``google_genai_use_vertexai`` is true and ``google_cloud_project``
    is set; otherwise Gemini Developer API with ``google_api_key``.
    """
    s = settings or get_settings()
    key = api_key or s.google_api_key

    if s.google_genai_use_vertexai and s.google_cloud_project:
        kwargs: dict = {
            "model": model,
            "temperature": temperature,
            "vertexai": True,
            "project": s.google_cloud_project,
            "location": s.google_cloud_location,
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
    )
