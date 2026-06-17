from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatSession(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

