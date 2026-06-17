from app.models.chat import ChatMessage
from app.schemas.common import SuccessResponse


class SessionResponse(SuccessResponse):
    session_id: str
    chat_history: list[ChatMessage]
    metadata: dict[str, str]

