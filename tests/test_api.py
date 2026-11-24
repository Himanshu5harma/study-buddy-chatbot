from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.chat_service import ChatService
from app.config import Settings
from app.main import create_app


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_answer_and_saves_history(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "What is photosynthesis?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Photosynthesis turns light into sugar."

    history = client.get("/sessions/s1/history").json()["messages"]
    assert history == [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis turns light into sugar."},
    ]


def test_stream_returns_full_text(client):
    r = client.post("/chat/stream", json={"session_id": "s2", "message": "hi"})
    assert r.status_code == 200
    assert r.text == "Photosynthesis turns light into sugar."
    assert len(client.get("/sessions/s2/history").json()["messages"]) == 2


def test_sessions_are_isolated_and_clearable(client):
    client.post("/chat", json={"session_id": "a", "message": "hi"})
    assert client.get("/sessions/b/history").json()["messages"] == []

    assert client.delete("/sessions/a").status_code == 204
    assert client.get("/sessions/a/history").json()["messages"] == []


def test_validation(client):
    assert client.post("/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "ok", "message": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "ok", "message": "x" * 101}).status_code == 413


def test_llm_failure_returns_502_and_saves_nothing(store):
    def boom(_):
        raise RuntimeError("provider down")

    svc = ChatService(RunnableLambda(boom), store, max_history_tokens=3000)
    with TestClient(create_app(Settings(openrouter_api_key="k"), chat_service=svc)) as c:
        r = c.post("/chat", json={"session_id": "x", "message": "hi"})
        assert r.status_code == 502
        assert c.get("/sessions/x/history").json()["messages"] == []


def test_history_is_trimmed_to_token_budget(fake_model, store):
    history = store.history("long")
    for i in range(50):
        history.add_messages([HumanMessage(f"question {i} " * 20), AIMessage(f"answer {i} " * 20)])

    svc = ChatService(fake_model, store, max_history_tokens=300)
    prepared = svc._prepare("long", "next")
    assert 0 < len(prepared["history"]) < 100
    assert isinstance(prepared["history"][0], HumanMessage)  # trimming starts on a human turn
    assert "answer 49" in prepared["history"][-1].content  # the most recent turns are kept
