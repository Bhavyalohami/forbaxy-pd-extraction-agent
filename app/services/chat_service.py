from app.agents.base import AgentPort
from app.models.chat import ChatMessage
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.memory import SessionStore
from app.services.privacy import redact_patient_information


class ChatService:
    def __init__(self, agent: AgentPort, session_store: SessionStore) -> None:
        self._agent = agent
        self._session_store = session_store

    async def chat(self, request: ChatRequest) -> ChatResponse:
        message = redact_patient_information(request.message)
        await self._session_store.append(
            request.session_id,
            ChatMessage(role="user", content=message),
        )
        agent_result = await self._agent.run(session_id=request.session_id, message=message)
        await self._session_store.append(
            request.session_id,
            ChatMessage(role="assistant", content=agent_result.response),
        )
        return ChatResponse(
            response=agent_result.response,
            sources=agent_result.sources,
            session_id=request.session_id,
        )
