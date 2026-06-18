from app.agents.prescription_agent import PrescriptionAgent
from app.schemas.pd import LearningContext


def test_learning_context_is_sanitized_and_formatted():
    formatted = PrescriptionAgent.format_learning_context(
        LearningContext(
            guidance="Verify dosage. Patient name: Hidden",
            patterns=["HTN means hypertension"],
            common_corrections=["Mobile 9999999999 should not appear"],
        )
    )

    assert "LEARNING CONTEXT" in formatted
    assert "HTN means hypertension" in formatted
    assert "Hidden" not in formatted
    assert "Patient name" not in formatted
    assert "9999999999" not in formatted


def test_learning_context_does_not_include_raw_reviewed_output_shape():
    formatted = PrescriptionAgent.format_learning_context(
        LearningContext(common_corrections=['{"pd_extraction":{"diagnosis":"x"}}'])
    )

    assert "reviewed_output" not in formatted
    assert "source_hash" not in formatted
