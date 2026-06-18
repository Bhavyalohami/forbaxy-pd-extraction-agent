import json

from app.schemas.pd import (
    LearningContext,
    LearningMetadata,
    ProductionPatientSummary,
    ProductionPDResponse,
)
from app.schemas.prescription import PDExtractionResponse
from app.services.privacy import redact_patient_information


def percent(value: float) -> int:
    return max(0, min(100, round(value * 100)))


def build_learning_metadata(
    *,
    learning_used: bool,
    extraction_id: str | None,
    supplied_metadata: LearningMetadata | None,
) -> dict[str, int | float | str | bool | None]:
    metadata: dict[str, int | float | str | bool | None] = {
        "learning_used": learning_used,
        "retrieval_matches": 0,
        "average_similarity": None,
        "context_size": 0,
        "extraction_id": None,
    }
    if supplied_metadata is not None:
        metadata.update(
            {
                key: value
                for key, value in supplied_metadata.model_dump(exclude_none=True).items()
                if key != "learning_used" and _metadata_value_is_safe(value)
            }
        )
    # Response truth is computed by the agent. Caller-supplied learning_used is never echoed,
    # because learning is considered used only when non-empty guidance was actually injected.
    metadata["learning_used"] = learning_used
    if extraction_id and _metadata_value_is_safe(extraction_id):
        metadata["extraction_id"] = extraction_id
    return metadata


def learning_context_has_content(learning_context: LearningContext | None) -> bool:
    if learning_context is None:
        return False
    return any(
        (
            bool(redact_patient_information(learning_context.guidance).strip()),
            any(redact_patient_information(item).strip() for item in learning_context.patterns),
            any(
                redact_patient_information(item).strip()
                for item in learning_context.common_corrections
            ),
        )
    )


def to_production_response(
    extraction_response: PDExtractionResponse,
    *,
    learning_metadata: dict[str, int | float | str | bool | None],
) -> ProductionPDResponse:
    extraction = extraction_response.pd_extraction
    clinical_notes = "\n".join(
        item
        for item in (
            extraction.history,
            extraction.examination,
            extraction.preventive_advice,
            extraction.notes,
        )
        if item
    )
    return ProductionPDResponse(
        confidence_score=percent(extraction.extraction_confidence),
        ipd_probability=percent(extraction.admission.ipd_probability),
        admission_advised=extraction.admission.advised,
        follow_up_date=extraction.follow_up.date or None,
        consultant=extraction.consultant,
        patient=ProductionPatientSummary(
            diagnosis=redact_patient_information(extraction.diagnosis),
            clinical_notes=redact_patient_information(clinical_notes),
            follow_up=redact_patient_information(extraction.follow_up.instruction),
        ),
        vitals=extraction.vitals,
        medicines=extraction.medicines,
        investigations=extraction.investigations,
        medication_assessment=extraction.medication_assessment,
        issues=[redact_patient_information(item) for item in extraction.unclear_fields],
        learning_metadata=learning_metadata,
    )


def parse_internal_extraction(response: str) -> PDExtractionResponse:
    payload = json.loads(response)
    return PDExtractionResponse.model_validate(payload)


def _metadata_value_is_safe(value: object) -> bool:
    if isinstance(value, str):
        return redact_patient_information(value) == value
    return True
