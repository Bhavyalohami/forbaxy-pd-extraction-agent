import json
import re
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from openai import AsyncOpenAI, AuthenticationError, OpenAIError
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
        api_key: str,
        api_base: str,
        model_name: str,
        vision_model_name: str,
        temperature: float,
        enable_vision_input: bool,
    ) -> None:
        self._session_store = session_store
        self._review_learning_store = review_learning_store
        self._llm = llm
        self._api_key = api_key
        self._api_base = api_base
        self._model_name = model_name
        self._vision_model_name = vision_model_name or model_name
        self._temperature = temperature
        self._enable_vision_input = enable_vision_input
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
        image_data_url: str | None = None,
    ) -> AgentResult:
        logger.info("agent_execution_started", extra={"session_id": session_id})
        ctx = self._contexts.setdefault(session_id, Context(self._agent))
        learning_context_used = learning_context_has_content(learning_context)
        injected_message = self._inject_context(
            message,
            learning_context=learning_context if learning_context_used else None,
        )
        try:
            if image_data_url and self._enable_vision_input:
                response = await self._complete_with_image(injected_message, image_data_url)
            elif self._direct_llm_mode:
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
                "Prescription extraction model failed.",
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

    async def _complete_with_image(self, message: str, image_data_url: str) -> str:
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._api_base)
        response = await client.chat.completions.create(
            model=self._vision_model_name,
            temperature=self._temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{PD_SYSTEM_PROMPT}\n\n{message}"},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""

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
            payload = self._normalise_llm_payload(json.loads(candidate))
            return PDExtractionResponse.model_validate(payload).model_dump_json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AppError(
                "Prescription extraction model failed.",
                error_code="LLM_PROVIDER_ERROR",
                status_code=502,
            ) from exc

    def _normalise_llm_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        if "pd_extraction" not in payload:
            payload = {"pd_extraction": payload}

        extraction = payload.get("pd_extraction")
        if not isinstance(extraction, dict):
            return payload

        extraction["investigations"] = self._normalise_investigations(
            extraction.get("investigations", [])
        )
        extraction["medicines"] = self._normalise_medicines(extraction.get("medicines", []))
        extraction["admission"] = self._normalise_admission(extraction.get("admission", {}))
        extraction["medication_assessment"] = self._normalise_medication_assessment(
            extraction.get("medication_assessment", {})
        )
        extraction["extraction_confidence"] = self._fraction(
            extraction.get("extraction_confidence", 0)
        )
        payload["pd_extraction"] = extraction
        return payload

    def _normalise_investigations(self, investigations: Any) -> list[dict[str, str]]:
        if not isinstance(investigations, list):
            return []

        normalised = []
        for investigation in investigations:
            if isinstance(investigation, str):
                name = investigation.strip()
                if name:
                    normalised.append({"name": name, "notes": "", "status": "ordered"})
                continue
            if isinstance(investigation, dict):
                status = str(investigation.get("status", "ordered") or "ordered")
                if status not in {"ordered", "pending", "completed", "cancelled", "unknown"}:
                    status = "ordered"
                normalised.append(
                    {
                        "name": str(investigation.get("name", "") or ""),
                        "notes": str(investigation.get("notes", "") or ""),
                        "status": status,
                    }
                )
        return normalised

    def _normalise_medicines(self, medicines: Any) -> list[dict[str, Any]]:
        if not isinstance(medicines, list):
            return []

        normalised = []
        for medicine in medicines:
            if isinstance(medicine, str):
                name = medicine.strip()
                if name:
                    normalised.append({"name": name, "confidence": 0})
                continue
            if isinstance(medicine, dict):
                item = dict(medicine)
                item["confidence"] = self._fraction(item.get("confidence", 0))
                normalised.append(item)
        return normalised

    def _normalise_admission(self, admission: Any) -> dict[str, Any]:
        if not isinstance(admission, dict):
            return {}
        item = dict(admission)
        item["ipd_probability"] = self._fraction(item.get("ipd_probability", 0))
        return item

    def _normalise_medication_assessment(self, assessment: Any) -> dict[str, Any]:
        if not isinstance(assessment, dict):
            return {}

        item = dict(assessment)
        item["confidence"] = self._fraction(item.get("confidence", 0))
        return item

    def _fraction(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0
        if number > 1:
            number /= 100
        return max(0, min(1, number))

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
