from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prescription import Investigation, MedicationAssessment, Medicine, Vitals

ContentType = Literal["text", "image", "document"]


class LearningContext(BaseModel):
    guidance: str = ""
    patterns: list[str] = Field(default_factory=list)
    common_corrections: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LearningMetadata(BaseModel):
    learning_used: bool | None = None
    retrieval_matches: int | None = Field(default=None, ge=0)
    average_similarity: float | None = Field(default=None, ge=0, le=1)
    context_size: int | None = Field(default=None, ge=0)
    extraction_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class PDExtractRequest(BaseModel):
    content: str = Field(min_length=1)
    content_type: ContentType = "text"
    learning_context: LearningContext | None = None
    learning_metadata: LearningMetadata | None = None
    extraction_id: str | None = None
    prescription_id: str | None = None
    provider: str | None = None
    model: str | None = None

    model_config = ConfigDict(extra="forbid")


class ProductionPatientSummary(BaseModel):
    diagnosis: str = ""
    clinical_notes: str = ""


class ProductionPDResponse(BaseModel):
    confidence_score: int = Field(default=0, ge=0, le=100)
    ipd_probability: int = Field(default=0, ge=0, le=100)
    patient: ProductionPatientSummary = Field(default_factory=ProductionPatientSummary)
    vitals: Vitals = Field(default_factory=Vitals)
    medicines: list[Medicine] = Field(default_factory=list)
    investigations: list[Investigation] = Field(default_factory=list)
    medication_assessment: MedicationAssessment = Field(default_factory=MedicationAssessment)
    learning_metadata: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
