from app.schemas.pd import LearningContext, LearningMetadata
from app.schemas.prescription import AdmissionAdvice, PDExtraction, PDExtractionResponse
from app.services.production_adapter import (
    build_learning_metadata,
    learning_context_has_content,
    normalise_issues,
    percent,
    to_production_response,
)


def test_percent_converts_fraction_to_production_score():
    assert percent(0) == 0
    assert percent(0.95) == 95
    assert percent(1) == 100


def test_adapter_converts_confidence_and_ipd_probability():
    response = PDExtractionResponse(
        pd_extraction=PDExtraction(
            diagnosis="viral fever",
            extraction_confidence=0.95,
            admission=AdmissionAdvice(ipd_probability=0.3),
        )
    )

    production = to_production_response(
        response,
        learning_metadata={"learning_used": False},
    )

    assert production.confidence_score == 95
    assert production.ipd_probability == 30
    assert production.admission_advised is False
    assert production.follow_up_date is None
    assert production.patient.follow_up == ""
    assert production.vitals.rbs == ""
    assert production.issues == []


def test_learning_metadata_does_not_echo_pi_like_values():
    metadata = build_learning_metadata(
        learning_used=True,
        extraction_id="patient name: hidden",
        supplied_metadata=LearningMetadata(retrieval_matches=2),
    )

    assert metadata == {
        "learning_used": True,
        "retrieval_matches": 2,
        "average_similarity": None,
        "context_size": 0,
        "extraction_id": None,
    }


def test_learning_metadata_does_not_trust_caller_learning_used():
    metadata = build_learning_metadata(
        learning_used=False,
        extraction_id=None,
        supplied_metadata=LearningMetadata(learning_used=True, retrieval_matches=2),
    )

    assert metadata == {
        "learning_used": False,
        "retrieval_matches": 2,
        "average_similarity": None,
        "context_size": 0,
        "extraction_id": None,
    }


def test_learning_context_has_content_only_for_non_empty_context():
    assert learning_context_has_content(None) is False
    assert learning_context_has_content(LearningContext()) is False
    assert (
        learning_context_has_content(LearningContext(guidance="Prefer visible evidence."))
        is True
    )


def test_normalise_issues_removes_optional_raw_field_names():
    assert normalise_issues(["diagnosis", "pain_score", "height", "rbs"]) == [
        "Diagnosis is not clearly visible."
    ]


def test_adapter_returns_human_readable_issues():
    response = PDExtractionResponse(
        pd_extraction=PDExtraction(
            unclear_fields=["diagnosis", "pain_score", "route", "duration"]
        )
    )

    production = to_production_response(
        response,
        learning_metadata={"learning_used": False},
    )

    assert production.issues == [
        "Diagnosis is not clearly visible.",
        "Medicine route is not specified.",
        "Medicine duration is not clearly visible.",
    ]
