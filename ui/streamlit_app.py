"""Streamlit chat UI. It only talks to the FastAPI backend over HTTP.

Keeping the UI separate from the LLM logic means you could replace it with
React, a mobile app, or a Slack bot without touching the backend.

Run:  streamlit run ui/streamlit_app.py
"""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Study Buddy", page_icon="📚")
st.title("📚 Study Buddy")
st.caption("Ask me to explain a topic, or say *quiz me on ...*")

# A session id identifies one conversation. Keep it in the URL so a page
# refresh keeps the same conversation (the backend stores the history).
if "sid" not in st.query_params:
    st.query_params["sid"] = uuid.uuid4().hex[:16]
session_id = st.query_params["sid"]


def load_history() -> list[dict]:
    try:
        r = httpx.get(f"{API_URL}/sessions/{session_id}/history", timeout=10)
        r.raise_for_status()
        return r.json()["messages"]
    except httpx.HTTPError:
        st.error(f"Can't reach the backend at {API_URL}. Is `uvicorn app.main:app` running?")
        return []


def stream_answer(message: str):
    with httpx.stream(
        "POST",
        f"{API_URL}/chat/stream",
        json={"session_id": session_id, "message": message},
        timeout=httpx.Timeout(120, connect=10),
    ) as r:
        if r.status_code != 200:
            r.read()
            yield f"⚠️ Error {r.status_code}: {r.json().get('detail', r.text)}"
            return
        yield from r.iter_text()


with st.sidebar:
    st.subheader("Conversation")
    st.code(session_id, language=None)
    if st.button("🗑️ New conversation", use_container_width=True):
        httpx.delete(f"{API_URL}/sessions/{session_id}", timeout=10)
        st.query_params["sid"] = uuid.uuid4().hex[:16]
        st.rerun()

for msg in load_history():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What do you want to learn today?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            st.write_stream(stream_answer(prompt))
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
