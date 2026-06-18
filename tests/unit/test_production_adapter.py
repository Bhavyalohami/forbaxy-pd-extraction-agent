from app.schemas.pd import LearningMetadata
from app.schemas.prescription import AdmissionAdvice, PDExtraction, PDExtractionResponse
from app.services.production_adapter import build_learning_metadata, percent, to_production_response


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


def test_learning_metadata_does_not_echo_pi_like_values():
    metadata = build_learning_metadata(
        learning_used=True,
        extraction_id="patient name: hidden",
        supplied_metadata=LearningMetadata(retrieval_matches=2),
    )

    assert metadata == {"learning_used": True, "retrieval_matches": 2}
