import base64
import re
from typing import Any
from uuid import uuid4

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from openai import AsyncOpenAI, OpenAIError

from app.agents.prescription_agent import PrescriptionAgent
from app.agents.prompts import PD_SYSTEM_PROMPT
from app.config.settings import Settings
from app.core.exceptions import AppError
from app.parsers.factory import build_parser
from app.schemas.pd import LearningContext, LearningMetadata, PDExtractRequest
from app.schemas.prescription import PDExtractionResponse
from app.services.privacy import contains_patient_information, redact_patient_information
from app.services.production_adapter import (
    build_learning_metadata,
    learning_context_has_content,
    to_production_response,
)


class PDExtractionWorkflow(Workflow):
    """Deployable LlamaAgents workflow for cropped PD-only extraction."""

    @step()
    async def extract(self, ev: StartEvent) -> StopEvent:
        request = self._request_from_event(ev)
        settings = Settings()
        content = request.content
        image_data_url = None
        if request.content_type in {"image", "document"} and _looks_like_raw_file_payload(content):
            if request.content_type == "image":
                image_data_url = _normalise_image_data_url(content)
            parsed = await build_parser(settings).parse(
                content=_decode_raw_content(content),
                filename=f"pd-crop.{_default_extension(content, request.content_type)}",
                content_type=_content_mime_type(content, request.content_type),
            )
            content = parsed.text

        if contains_patient_information(content):
            raise AppError(
                "Patient information is not accepted by this endpoint.",
                error_code="PRIVACY_BOUNDARY",
            )

        learning_context_used = learning_context_has_content(request.learning_context)
        learning_context = request.learning_context if learning_context_used else None
        prompt = (
            f"{PD_SYSTEM_PROMPT}\n\n"
            "Context: Input is expected to contain only cropped Prescription Details. "
            "Return only the required JSON object.\n\n"
            f"{PrescriptionAgent.format_learning_context(learning_context)}\n\n"
            f"Task input:\n{redact_patient_information(content)}"
        )

        try:
            response = await _complete_with_chat(
                prompt=prompt,
                image_data_url=image_data_url if settings.enable_vision_input else None,
                settings=settings,
            )
        except OpenAIError as exc:
            raise AppError(
                f"Prescription extraction model failed: {_safe_provider_error(exc)}",
                error_code="LLM_PROVIDER_ERROR",
                status_code=502,
            ) from exc

        try:
            json_text = PrescriptionAgent.extract_json_object(
                redact_patient_information(str(response))
            )
            extraction = PDExtractionResponse.model_validate_json(json_text)
        except ValueError as exc:
            raise AppError(
                "Prescription extraction model failed.",
                error_code="LLM_PROVIDER_ERROR",
                status_code=502,
            ) from exc

        production_response = to_production_response(
            extraction,
            learning_metadata=build_learning_metadata(
                learning_used=learning_context_used,
                extraction_id=request.extraction_id or str(uuid4()),
                supplied_metadata=request.learning_metadata,
            ),
        )
        return StopEvent(result=production_response.model_dump())

    def _request_from_event(self, ev: StartEvent) -> PDExtractRequest:
        payload = self._payload_from_event(ev)
        if isinstance(payload, dict):
            return PDExtractRequest.model_validate(payload)
        return PDExtractRequest(content=str(payload), content_type="text")

    def _payload_from_event(self, ev: StartEvent) -> dict[str, Any] | str:
        data = ev.get("data")
        if isinstance(data, dict):
            return data

        content = self._message_from_event(ev)
        payload: dict[str, Any] = {
            "content": content,
            "content_type": ev.get("content_type", "text"),
            "extraction_id": ev.get("extraction_id"),
            "prescription_id": ev.get("prescription_id"),
            "provider": ev.get("provider"),
            "model": ev.get("model"),
        }
        learning_context = ev.get("learning_context")
        if isinstance(learning_context, dict) or isinstance(learning_context, LearningContext):
            payload["learning_context"] = learning_context
        learning_metadata = ev.get("learning_metadata")
        if isinstance(learning_metadata, dict) or isinstance(learning_metadata, LearningMetadata):
            payload["learning_metadata"] = learning_metadata
        return {key: value for key, value in payload.items() if value is not None}

    def _message_from_event(self, ev: StartEvent) -> str:
        for key in ("content", "message", "query", "input", "text"):
            value = ev.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(ev.get("data", ""))


def _looks_like_raw_file_payload(content: str) -> bool:
    stripped = content.strip()
    if stripped.startswith("data:") or ";base64," in stripped[:120]:
        return True
    compact = re.sub(r"\s+", "", stripped)
    return len(compact) >= 200 and len(compact) % 4 == 0 and re.fullmatch(
        r"[A-Za-z0-9+/]+={0,2}",
        compact,
    ) is not None


def _decode_raw_content(content: str) -> bytes:
    stripped = content.strip()
    if stripped.startswith("data:") and "," in stripped:
        stripped = stripped.split(",", 1)[1]
    compact = re.sub(r"\s+", "", stripped)
    try:
        return base64.b64decode(compact, validate=True)
    except Exception as exc:
        raise AppError(
            "Invalid base64 image/document payload.",
            error_code="INVALID_PD_PAYLOAD",
            status_code=422,
        ) from exc


async def _complete_with_chat(
    *,
    prompt: str,
    image_data_url: str | None,
    settings: Settings,
) -> str:
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
    content: str | list[dict[str, object]]
    if image_data_url:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        content = prompt
    response = await client.chat.completions.create(
        model=settings.model_name,
        temperature=settings.temperature,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content or ""


def _normalise_image_data_url(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("data:"):
        return stripped
    compact = re.sub(r"\s+", "", stripped)
    return f"data:image/jpeg;base64,{compact}"


def _content_mime_type(content: str, content_type: str) -> str:
    match = re.match(r"^data:([^;,]+)", content.strip())
    if match:
        return match.group(1)
    return "application/pdf" if content_type == "document" else "image/jpeg"


def _default_extension(content: str, content_type: str) -> str:
    mime_type = _content_mime_type(content, content_type).lower()
    return {
        "image/png": "png",
        "image/webp": "webp",
        "application/pdf": "pdf",
    }.get(mime_type, "jpg" if content_type == "image" else "bin")


def _safe_provider_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message[:500] or type(exc).__name__


pd_extraction_workflow = PDExtractionWorkflow(timeout=45)
