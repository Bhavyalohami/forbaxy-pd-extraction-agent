import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.prescription import (
    Consultant,
    Investigation,
    MedicationAssessment,
    Medicine,
    Vitals,
)

ContentType = Literal["text", "image", "document"]


class LearningContext(BaseModel):
    guidance: str = ""
    patterns: list[str] = Field(default_factory=list)
    common_corrections: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LearningMetadata(BaseModel):
    learning_used: bool | None = Field(
        default=None,
        description=(
            "Request-side hint only. Response learning_used is computed by the agent and is true "
            "only when non-empty learning context is actually injected into the prompt."
        ),
    )
    retrieval_matches: int | None = Field(default=None, ge=0)
    average_similarity: float | None = Field(default=None, ge=0, le=1)
    context_size: int | None = Field(default=None, ge=0)
    extraction_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class PDExtractRequest(BaseModel):
    content: str = Field(
        min_length=1,
        description=(
            "PD-only clinical text. For content_type=image or document, this must still be "
            "already-extracted text/markdown from an upstream parser, not raw binary, base64, "
            "or a data URL."
        ),
    )
    content_type: ContentType = Field(
        default="text",
        description=(
            "Input label. The /pd/extract endpoint extracts from text only; image/document "
            "payloads must be converted to PD text before calling this endpoint."
        ),
    )
    learning_context: LearningContext | None = None
    learning_metadata: LearningMetadata | None = None
    extraction_id: str | None = None
    prescription_id: str | None = None
    provider: str | None = None
    model: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_raw_file_payloads(self) -> "PDExtractRequest":
        if (
            self.content_type in {"image", "document"}
            and _looks_like_raw_file_payload(self.content)
        ):
            raise ValueError(
                "content_type=image/document requires already-extracted PD text; raw base64, "
                "data URLs, and binary-like payloads are not accepted by /pd/extract."
            )
        return self


class ProductionPatientSummary(BaseModel):
    diagnosis: str = ""
    clinical_notes: str = ""
    follow_up: str = ""


class ProductionLearningMetadata(BaseModel):
    # True only when non-empty learning context was actually injected into the prompt.
    # Empty, disabled, rejected, or unavailable learning context must serialize as false.
    learning_used: bool = False
    retrieval_matches: int | None = Field(default=None, ge=0)
    average_similarity: float | None = Field(default=None, ge=0, le=1)
    context_size: int | None = Field(default=None, ge=0)
    extraction_id: str | None = None

    model_config = ConfigDict(extra="allow")


class ProductionPDResponse(BaseModel):
    confidence_score: int = Field(default=0, ge=0, le=100)
    ipd_probability: int = Field(default=0, ge=0, le=100)
    admission_advised: bool = False
    follow_up_date: str | None = None
    consultant: Consultant = Field(default_factory=Consultant)
    patient: ProductionPatientSummary = Field(default_factory=ProductionPatientSummary)
    vitals: Vitals = Field(default_factory=Vitals)
    medicines: list[Medicine] = Field(default_factory=list)
    investigations: list[Investigation] = Field(default_factory=list)
    medication_assessment: MedicationAssessment = Field(default_factory=MedicationAssessment)
    issues: list[str] = Field(default_factory=list)
    learning_metadata: ProductionLearningMetadata = Field(
        default_factory=ProductionLearningMetadata
    )


def _looks_like_raw_file_payload(content: str) -> bool:
    stripped = content.strip()
    if stripped.startswith("data:") or ";base64," in stripped[:120]:
        return True
    if len(stripped) < 200:
        return False
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 200 or len(compact) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact) is not None
