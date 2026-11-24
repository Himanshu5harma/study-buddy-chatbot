"""Request/response models for the API (validated by Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field

# Letters, digits, dash, underscore; stops odd values from ending up as keys.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN, examples=["student-123"])
    message: str = Field(..., min_length=1, examples=["Explain photosynthesis simply"])


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
