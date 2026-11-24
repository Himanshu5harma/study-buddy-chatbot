"""Application settings.

All configuration comes from environment variables (or a `.env` file), so no
secrets or environment-specific values are hard-coded. pydantic-settings
validates types at startup: a missing API key fails fast with a clear error
instead of failing on the first chat request.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider (OpenRouter is OpenAI-compatible) ---
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Free models end in ":free". Availability changes, so check
    # https://openrouter.ai/models?q=free and update these in .env if needed.
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    # Used automatically if the primary model errors (rate limits are common on free tiers).
    llm_fallback_model: str | None = "mistralai/mistral-small-3.2-24b-instruct:free"
    llm_temperature: float = Field(0.4, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, gt=0)
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # --- Conversation memory ---
    database_path: str = "data/chat_history.db"
    # Older messages are trimmed so the prompt stays inside the model's context window.
    max_history_tokens: int = 3000

    # --- API ---
    max_message_chars: int = 4000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501"]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment once)."""
    return Settings()
