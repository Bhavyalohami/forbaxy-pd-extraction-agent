from dataclasses import dataclass

from llama_index.llms.openai_like import OpenAILike

from app.agents.prescription_agent import PrescriptionAgent
from app.config.settings import Settings
from app.parsers.mock_parser import MockParser
from app.services.chat_service import ChatService
from app.services.memory import InMemorySessionStore
from app.services.review_learning import InMemoryReviewLearningStore
from app.tools.registry import ToolRegistry, build_default_registry


@dataclass(slots=True)
class Container:
    settings: Settings
    session_store: InMemorySessionStore
    review_learning_store: InMemoryReviewLearningStore
    tool_registry: ToolRegistry
    parser: MockParser
    agent: PrescriptionAgent
    chat_service: ChatService

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        session_store = InMemorySessionStore()
        review_learning_store = InMemoryReviewLearningStore()
        tool_registry = build_default_registry()
        parser = MockParser()
        llm = OpenAILike(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.temperature,
            context_window=settings.llm_context_window,
            is_chat_model=True,
            is_function_calling_model=settings.llm_is_function_calling,
        )
        agent = PrescriptionAgent(
            llm=llm,
            tool_registry=tool_registry,
            session_store=session_store,
            review_learning_store=review_learning_store,
            timeout_seconds=settings.agent_timeout_seconds,
        )
        chat_service = ChatService(agent=agent, session_store=session_store)
        return cls(
            settings=settings,
            session_store=session_store,
            review_learning_store=review_learning_store,
            tool_registry=tool_registry,
            parser=parser,
            agent=agent,
            chat_service=chat_service,
        )
