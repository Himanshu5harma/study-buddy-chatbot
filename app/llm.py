"""Chat model factory.

LangChain concept: a *chat model* is a Runnable that takes a list of messages
and returns an AIMessage. Every provider (OpenAI, Anthropic, Ollama, ...) has
its own class, but they all share the same interface (`invoke`, `stream`,
`ainvoke`, `astream`), so the rest of the app doesn't care which one we use.

OpenRouter speaks the OpenAI API, so we reuse `ChatOpenAI` and point its
`base_url` at OpenRouter. To switch providers later you only change this file.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings


def _make_model(settings: Settings, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,  # retries on 429 / 5xx / network errors
        streaming=True,
        # Optional headers OpenRouter uses to identify your app on its dashboard.
        default_headers={"X-Title": "Study Buddy Chatbot"},
    )


def build_chat_model(settings: Settings) -> BaseChatModel | Runnable:
    """Primary model, with an automatic fallback model if one is configured.

    LangChain concept: `.with_fallbacks()` wraps a Runnable so that if it
    raises, the next Runnable in the list is tried with the same input.
    """
    primary = _make_model(settings, settings.llm_model)
    if settings.llm_fallback_model:
        return primary.with_fallbacks([_make_model(settings, settings.llm_fallback_model)])
    return primary
