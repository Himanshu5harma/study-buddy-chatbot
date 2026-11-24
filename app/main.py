"""FastAPI backend.

Endpoints
    GET    /health                         liveness check
    POST   /chat                           full answer as JSON
    POST   /chat/stream                    answer streamed as plain text chunks
    GET    /sessions/{session_id}/history  conversation so far
    DELETE /sessions/{session_id}          forget a conversation

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.chat_service import ChatService
from app.config import Settings, get_settings
from app.llm import build_chat_model
from app.memory import SQLiteStore
from app.schemas import SESSION_ID_PATTERN, ChatRequest, ChatResponse, HistoryMessage, HistoryResponse

logger = logging.getLogger("study_buddy")


def create_app(settings: Settings | None = None, chat_service: ChatService | None = None) -> FastAPI:
    """App factory. Tests pass in a fake ChatService; production builds the real one."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = None
        if chat_service is None:
            store = SQLiteStore(settings.database_path)
            app.state.chat = ChatService(build_chat_model(settings), store, settings.max_history_tokens)
            logger.info("Started with model=%s fallback=%s", settings.llm_model, settings.llm_fallback_model)
        else:
            app.state.chat = chat_service
        yield
        if store:
            store.close()

    app = FastAPI(title="Study Buddy API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def service(request: Request) -> ChatService:
        return request.app.state.chat

    def check_length(req: ChatRequest) -> ChatRequest:
        if len(req.message) > settings.max_message_chars:
            raise HTTPException(413, f"Message too long (max {settings.max_message_chars} characters).")
        return req

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest = Depends(check_length), svc: ChatService = Depends(service)):
        try:
            answer = await svc.chat(req.session_id, req.message)
        except Exception:
            logger.exception("LLM call failed for session=%s", req.session_id)
            raise HTTPException(502, "The AI model is unavailable right now. Please try again.")
        return ChatResponse(session_id=req.session_id, answer=answer)

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest = Depends(check_length), svc: ChatService = Depends(service)):
        async def generate() -> AsyncIterator[str]:
            try:
                async for chunk in svc.stream(req.session_id, req.message):
                    yield chunk
            except Exception:
                # Headers are already sent once streaming starts, so we can't
                # change the status code; send a readable error marker instead.
                logger.exception("Streaming failed for session=%s", req.session_id)
                yield "\n\n[error] The AI model is unavailable right now. Please try again."

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

    @app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
    async def history(
        session_id: str = Path(..., pattern=SESSION_ID_PATTERN), svc: ChatService = Depends(service)
    ):
        messages = [
            HistoryMessage(role="user" if isinstance(m, HumanMessage) else "assistant", content=str(m.content))
            for m in svc.get_history(session_id)
        ]
        return HistoryResponse(session_id=session_id, messages=messages)

    @app.delete("/sessions/{session_id}", status_code=204)
    async def clear(session_id: str = Path(..., pattern=SESSION_ID_PATTERN), svc: ChatService = Depends(service)):
        svc.clear_history(session_id)

    return app


app = create_app()  # uvicorn imports `app.main:app`
