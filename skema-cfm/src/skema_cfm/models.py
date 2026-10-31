"""LLM factory: one integration point to switch provider and model."""

from __future__ import annotations

from .config import RunConfig


def create_model(config: RunConfig):
    """Create a LangChain model via init_chat_model, never expose the secret in traces."""
    from langchain.chat_models import init_chat_model

    if not config.credential:
        raise ValueError(
            "Missing OPENROUTER_API_KEY; set .env or paste a key in Streamlit."
        )
    if config.backend == "openrouter":
        # Modern dedicated LangChain OpenRouter integration, provider/model slug unchanged.
        return init_chat_model(
            model=config.model,
            model_provider="openrouter",
            api_key=config.credential,
            temperature=0,
            max_retries=0,
        )
    # Optional compatibility bridge for older OpenRouter-compatible chat API models.
    return init_chat_model(
        model=config.model,
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=config.credential,
        temperature=0,
        max_retries=0,
    )
