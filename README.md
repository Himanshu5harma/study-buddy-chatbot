# 📚 Study Buddy: a LangChain chatbot

A tutor chatbot that explains topics, quizzes you and remembers the conversation.
It's built to learn **LangChain** properly, with a production-style structure:
FastAPI backend, Streamlit UI, persistent memory, streaming, retries/fallbacks, config and tests.

```
 Streamlit UI  ──HTTP──▶  FastAPI  ──▶  ChatService  ──▶  prompt | model | parser  ──▶  OpenRouter
 (ui/)                    (main.py)     (chat_service.py)        (LCEL chain)              (free LLMs)
                                            │
                                            └──▶ SQLite chat history (memory.py)
```

## Quick start (Windows PowerShell)

```powershell
cd study-buddy-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env      # then put your OpenRouter key in .env
```

Get a free key at <https://openrouter.ai/keys>. Then run the backend and the UI in **two terminals**:

```powershell
uvicorn app.main:app --reload          # API on http://localhost:8000  (docs: /docs)
```
```powershell
streamlit run ui/streamlit_app.py      # UI on http://localhost:8501
```

Run the tests (no API key or network needed; they use a fake LLM):

```powershell
pytest
```

## Project layout: where each LangChain concept lives

| File | What it does | LangChain concept |
|---|---|---|
| `app/config.py` | Settings from `.env`, validated at startup | n/a |
| `app/llm.py` | Creates the chat model (OpenRouter via `ChatOpenAI`) | **Chat models**, `.with_fallbacks()`, retries |
| `app/prompts.py` | The Study Buddy system prompt | **`ChatPromptTemplate`**, **`MessagesPlaceholder`** |
| `app/memory.py` | Stores messages per session in SQLite | **`BaseChatMessageHistory`**, message serialization |
| `app/chat_service.py` | Load history → trim → run chain → save | **LCEL** (`prompt \| model \| parser`), **`trim_messages`**, **streaming** (`astream`) |
| `app/main.py` | REST API + streaming endpoint | n/a |
| `ui/streamlit_app.py` | Chat UI, talks to the API over HTTP | n/a |
| `tests/` | API and memory tests | **Fake chat models** for testing |

## Things to try next (to learn more LangChain)

1. **Swap the model:** change `LLM_MODEL` in `.env`. No code changes needed.
2. **Change the personality:** edit `STUDY_BUDDY_SYSTEM_PROMPT` in `app/prompts.py`.
3. **Structured output:** use `model.with_structured_output(QuizQuestion)` to generate quiz questions as JSON.
4. **Use `RunnableWithMessageHistory`:** replace the manual load/save in `chat_service.py` with LangChain's helper.
5. **Tracing:** set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to see every chain step in LangSmith.
6. **Later:** RAG (chat with your notes) and agents (tools).

## Notes

- Free OpenRouter models have rate limits and come and go. If you see errors, pick another `:free`
  model at <https://openrouter.ai/models?q=free>. The fallback model is tried automatically.
- `.env` and `data/` are git-ignored, so your key and chat history never get committed.
