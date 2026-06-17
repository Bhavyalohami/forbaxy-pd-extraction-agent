from abc import ABC, abstractmethod
from copy import deepcopy

from app.core.exceptions import NotFoundError
from app.models.chat import ChatMessage, ChatSession


class SessionStore(ABC):
    @abstractmethod
    async def get_or_create(self, session_id: str) -> ChatSession:
        raise NotImplementedError

    @abstractmethod
    async def get(self, session_id: str) -> ChatSession:
        raise NotImplementedError

    @abstractmethod
    async def append(self, session_id: str, message: ChatMessage) -> ChatSession:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    async def get_or_create(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        return deepcopy(self._sessions[session_id])

    async def get(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            raise NotFoundError(f"Session '{session_id}' was not found.")
        return deepcopy(self._sessions[session_id])

    async def append(self, session_id: str, message: ChatMessage) -> ChatSession:
        session = await self.get_or_create(session_id)
        session.messages.append(message)
        self._sessions[session_id] = session
        return deepcopy(session)

    async def delete(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise NotFoundError(f"Session '{session_id}' was not found.")
        del self._sessions[session_id]

