import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# app.main builds a real app at import time; give it a dummy key so importing works.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from app.chat_service import ChatService  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.memory import SQLiteStore  # noqa: E402


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def fake_model():
    # A LangChain fake chat model: returns these replies in order, no network calls.
    return FakeListChatModel(responses=["Photosynthesis turns light into sugar.", "Yes, that's right!"])


@pytest.fixture
def service(fake_model, store):
    return ChatService(fake_model, store, max_history_tokens=3000)


@pytest.fixture
def client(service):
    settings = Settings(openrouter_api_key="test-key", max_message_chars=100)
    with TestClient(create_app(settings=settings, chat_service=service)) as c:
        yield c
