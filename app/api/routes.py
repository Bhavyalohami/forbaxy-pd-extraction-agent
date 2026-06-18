import base64
import re
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.core.container import Container
from app.core.dependencies import get_container
from app.core.exceptions import AppError
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import SuccessResponse
from app.schemas.documents import UploadResponse
from app.schemas.pd import PDExtractRequest, ProductionPDResponse
from app.schemas.sessions import SessionResponse
from app.services.privacy import contains_patient_information, redact_patient_information
from app.services.production_adapter import (
    build_learning_metadata,
    learning_context_has_content,
    parse_internal_extraction,
    to_production_response,
)

router = APIRouter()


@router.get("/health", response_model=SuccessResponse)
async def health() -> SuccessResponse:
    return SuccessResponse(message="healthy")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, container: Container = Depends(get_container)) -> ChatResponse:
    return await container.chat_service.chat(request)


@router.post(
    "/pd/extract",
    response_model=ProductionPDResponse,
)
async def extract_pd(
    request: PDExtractRequest,
    container: Container = Depends(get_container),
) -> ProductionPDResponse:
    content = request.content
    image_data_url = None
    if request.content_type in {"image", "document"} and _looks_like_raw_file_payload(content):
        if request.content_type == "image":
            image_data_url = _normalise_image_data_url(content)
        parsed = await container.parser.parse(
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

    learning_context = (
        request.learning_context if learning_context_has_content(request.learning_context) else None
    )
    agent_result = await container.agent.run(
        session_id=request.extraction_id or str(uuid4()),
        message=redact_patient_information(content),
        learning_context=learning_context,
        learning_metadata=request.learning_metadata,
        extraction_id=request.extraction_id,
        image_data_url=image_data_url,
    )
    return to_production_response(
        parse_internal_extraction(agent_result.response),
        learning_metadata=agent_result.learning_metadata
        or build_learning_metadata(
            learning_used=learning_context is not None,
            extraction_id=request.extraction_id,
            supplied_metadata=request.learning_metadata,
        ),
    )


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
) -> UploadResponse:
    content = await file.read()
    max_size = container.settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise AppError(
            "Uploaded file exceeds the configured size limit.",
            error_code="UPLOAD_TOO_LARGE",
        )
    parsed = await container.parser.parse(
        content=content,
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
    )
    return UploadResponse(
        document_id=str(uuid4()),
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        parsed_text=redact_patient_information(parsed.text),
        parser=container.parser.name,
        metadata=parsed.metadata,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    container: Container = Depends(get_container),
) -> SessionResponse:
    session = await container.session_store.get(session_id)
    return SessionResponse(
        message="session found",
        session_id=session.session_id,
        chat_history=session.messages,
        metadata=session.metadata,
    )


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: str,
    container: Container = Depends(get_container),
) -> SuccessResponse:
    await container.session_store.delete(session_id)
    return SuccessResponse(message="session deleted")


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
