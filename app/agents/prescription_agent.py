import json
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

from app.agents.base import AgentPort, AgentResult
from app.agents.prompts import PD_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.schemas.prescription import PDExtractionResponse
from app.services.memory import SessionStore
from app.services.privacy import redact_patient_information
from app.services.review_learning import ReviewLearningStore
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


class PrescriptionAgent(AgentPort):
    def __init__(
        self,
        *,
        llm: Any,
        tool_registry: ToolRegistry,
        session_store: SessionStore,
        review_learning_store: ReviewLearningStore,
    ) -> None:
        self._session_store = session_store
        self._review_learning_store = review_learning_store
        self._agent = FunctionAgent(
            name="forbaxy_pd_extraction_agent",
            description="PD-only prescription extraction, validation, and review-learning agent.",
            tools=tool_registry.as_llamaindex_tools(),
            llm=llm,
            system_prompt=PD_SYSTEM_PROMPT,
            initial_state={"future_mcp_enabled": False, "multi_agent_ready": True},
        )
        self._contexts: dict[str, Context] = {}

    async def run(self, session_id: str, message: str) -> AgentResult:
        logger.info("agent_execution_started", extra={"session_id": session_id})
        ctx = self._contexts.setdefault(session_id, Context(self._agent))
        examples = await self._review_learning_store.list_recent_examples(limit=3)
        injected_message = self._inject_context(
            message,
            examples=[example.reviewed_output for example in examples],
        )
        response = await self._agent.run(user_msg=injected_message, ctx=ctx)
        logger.info("agent_execution_completed", extra={"session_id": session_id})
        return AgentResult(response=self._coerce_pd_json(str(response)), sources=[])

    def _inject_context(self, message: str, examples: list[dict[str, Any]]) -> str:
        reviewed_context = redact_patient_information(json.dumps(examples[-3:], ensure_ascii=False))
        return (
            "Context: Input is expected to contain only cropped Prescription Details. "
            "Reviewed DMS-MS corrections are ground truth. Use these sanitized examples only for "
            f"format and abbreviation learning: {reviewed_context}\n\n"
            f"Task input:\n{message}"
        )

    def _coerce_pd_json(self, raw_response: str) -> str:
        sanitized = redact_patient_information(raw_response.strip())
        try:
            payload = json.loads(sanitized)
            return PDExtractionResponse.model_validate(payload).model_dump_json()
        except (json.JSONDecodeError, ValueError):
            fallback = PDExtractionResponse()
            fallback.pd_extraction.medication_assessment.status = "unclear"
            fallback.pd_extraction.medication_assessment.review_recommended = True
            fallback.pd_extraction.unclear_fields.append("agent_response")
            fallback.pd_extraction.notes = (
                "Agent response was not valid PD JSON and requires review."
            )
            return fallback.model_dump_json()
