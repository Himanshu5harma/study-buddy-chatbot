"""The chat service: where the LangChain pieces come together.

LangChain concept (LCEL): Runnables are composed with the `|` operator:

    prompt | model | StrOutputParser()

    dict of variables -> list of messages -> AIMessage chunks -> plain text

Memory is handled explicitly here (load history -> trim -> run chain -> save),
so you can see each step. LangChain also offers `RunnableWithMessageHistory`
and LangGraph checkpointers that automate this; they do the same thing.
"""

import logging
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from app.memory import SQLiteStore
from app.prompts import build_prompt

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, model: BaseChatModel | Runnable, store: SQLiteStore, max_history_tokens: int):
        self._store = store
        self._chain = build_prompt() | model | StrOutputParser()
        # Keep the most recent messages that fit the token budget. Trimming
        # always starts on a human message so the history stays well-formed.
        self._trimmer = trim_messages(
            max_tokens=max_history_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

    def get_history(self, session_id: str) -> list[BaseMessage]:
        return self._store.history(session_id).messages

    def clear_history(self, session_id: str) -> None:
        self._store.history(session_id).clear()

    def _prepare(self, session_id: str, user_input: str) -> dict:
        history = self._trimmer.invoke(self.get_history(session_id))
        return {"history": history, "input": user_input}

    def _save(self, session_id: str, user_input: str, answer: str) -> None:
        # Save only after a successful answer, so failed calls don't leave a
        # dangling question in the history.
        self._store.history(session_id).add_messages([HumanMessage(user_input), AIMessage(answer)])

    async def chat(self, session_id: str, user_input: str) -> str:
        """Return the full answer in one go."""
        answer = await self._chain.ainvoke(self._prepare(session_id, user_input))
        self._save(session_id, user_input, answer)
        logger.info("session=%s answered (%d chars)", session_id, len(answer))
        return answer

    async def stream(self, session_id: str, user_input: str) -> AsyncIterator[str]:
        """Yield the answer token-by-token as the model generates it."""
        parts: list[str] = []
        async for chunk in self._chain.astream(self._prepare(session_id, user_input)):
            parts.append(chunk)
            yield chunk
        answer = "".join(parts)
        if answer:
            self._save(session_id, user_input, answer)
        logger.info("session=%s streamed (%d chars)", session_id, len(answer))
