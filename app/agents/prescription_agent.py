import json
import re
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from openai import AuthenticationError, OpenAIError
from workflows.errors import WorkflowTimeoutError

from app.agents.base import AgentPort, AgentResult
from app.agents.prompts import PD_SYSTEM_PROMPT
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.pd import LearningContext, LearningMetadata
from app.schemas.prescription import PDExtractionResponse
from app.services.memory import SessionStore
from app.services.privacy import redact_patient_information
from app.services.production_adapter import build_learning_metadata, learning_context_has_content
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

    async def run(
        self,
        session_id: str,
        message: str,
        *,
        learning_context: LearningContext | None = None,
        learning_metadata: LearningMetadata | None = None,
        extraction_id: str | None = None,
    ) -> AgentResult:
        logger.info("agent_execution_started", extra={"session_id": session_id})
        ctx = self._contexts.setdefault(session_id, Context(self._agent))
        learning_context_used = learning_context_has_content(learning_context)
        injected_message = self._inject_context(
            message,
            learning_context=learning_context if learning_context_used else None,
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
        return AgentResult(
            response=self._coerce_pd_json(str(response)),
            sources=[],
            learning_metadata=build_learning_metadata(
                learning_used=learning_context_used,
                extraction_id=extraction_id,
                supplied_metadata=learning_metadata,
            ),
        )

    def _inject_context(
        self,
        message: str,
        *,
        learning_context: LearningContext | None = None,
    ) -> str:
        formatted_learning_context = self.format_learning_context(learning_context)
        return (
            "Context: Input is expected to contain only cropped Prescription Details. "
            "For ordinary PD extraction, do not call tools; return the required JSON directly. "
            f"{formatted_learning_context}\n\n"
            f"Task input:\n{message}"
        )

    @staticmethod
    def format_learning_context(learning_context: LearningContext | None) -> str:
        if learning_context is None:
            return "LEARNING CONTEXT\nNo external learning context supplied."

        lines = ["LEARNING CONTEXT"]
        guidance = redact_patient_information(learning_context.guidance.strip())
        if guidance:
            lines.extend(["", "Guidance:", guidance])

        patterns = _safe_items(learning_context.patterns)
        if patterns:
            lines.extend(["", "Patterns:"])
            lines.extend(f"- {item}" for item in patterns)

        common_corrections = _safe_items(learning_context.common_corrections)
        if common_corrections:
            lines.extend(["", "Common corrections:"])
            lines.extend(f"- {item}" for item in common_corrections)

        return "\n".join(lines)

    def _coerce_pd_json(self, raw_response: str) -> str:
        sanitized = redact_patient_information(raw_response.strip())
        candidate = self.extract_json_object(sanitized)
        try:
            payload = json.loads(candidate)
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

    @staticmethod
    def extract_json_object(response: str) -> str:
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        first = response.find("{")
        last = response.rfind("}")
        if first != -1 and last != -1 and last > first:
            return response[first : last + 1]

        return response


def _safe_items(items: list[str]) -> list[str]:
    return [redact_patient_information(item.strip()) for item in items if item.strip()]
