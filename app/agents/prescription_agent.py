import json
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from openai import AuthenticationError, OpenAIError
from workflows.errors import WorkflowTimeoutError

from app.agents.base import AgentPort, AgentResult
from app.agents.prompts import PD_SYSTEM_PROMPT
from app.core.exceptions import AppError
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
        timeout_seconds: float,
        enable_tools: bool,
        enable_structured_output: bool,
    ) -> None:
        self._session_store = session_store
        self._review_learning_store = review_learning_store
        self._llm = llm
        self._direct_llm_mode = not enable_tools and not enable_structured_output
        self._agent = FunctionAgent(
            name="forbaxy_pd_extraction_agent",
            description="PD-only prescription extraction, validation, and review-learning agent.",
            tools=tool_registry.as_llamaindex_tools() if enable_tools else [],
            llm=llm,
            system_prompt=PD_SYSTEM_PROMPT,
            initial_state={"future_mcp_enabled": False, "multi_agent_ready": True},
            output_cls=PDExtractionResponse if enable_structured_output else None,
            streaming=False,
            timeout=timeout_seconds,
            initial_tool_choice="auto" if enable_tools else "none",
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
        try:
            if self._direct_llm_mode:
                response = await self._llm.acomplete(f"{PD_SYSTEM_PROMPT}\n\n{injected_message}")
            else:
                response = await self._agent.run(user_msg=injected_message, ctx=ctx)
        except (TimeoutError, WorkflowTimeoutError) as exc:
            logger.warning("agent_execution_timeout", extra={"session_id": session_id})
            raise AppError(
                "Agent execution timed out.",
                error_code="AGENT_TIMEOUT",
                status_code=504,
            ) from exc
        except AuthenticationError as exc:
            logger.warning("llm_authentication_failed", extra={"session_id": session_id})
            raise AppError(
                "LLM authentication failed. Check OPENAI_API_KEY and OPENAI_API_BASE.",
                error_code="LLM_AUTHENTICATION_ERROR",
                status_code=503,
            ) from exc
        except OpenAIError as exc:
            logger.warning(
                "llm_provider_error",
                extra={"session_id": session_id, "provider_error": str(exc)},
            )
            raise AppError(
                "LLM provider request failed.",
                error_code="LLM_PROVIDER_ERROR",
                status_code=502,
            ) from exc
        logger.info("agent_execution_completed", extra={"session_id": session_id})
        return AgentResult(response=self._coerce_pd_json(str(response)), sources=[])

    def _inject_context(self, message: str, examples: list[dict[str, Any]]) -> str:
        reviewed_context = redact_patient_information(json.dumps(examples[-3:], ensure_ascii=False))
        return (
            "Context: Input is expected to contain only cropped Prescription Details. "
            "For ordinary PD extraction, do not call tools; return the required JSON directly. "
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
