from app.workflows.pd_extraction import PDExtractionWorkflow
from llama_index.core.workflow import StartEvent


def test_message_from_event_prefers_message_key():
    workflow = PDExtractionWorkflow()

    assert workflow._message_from_event(StartEvent(message="Rx text")) == "Rx text"


def test_request_from_event_accepts_production_payload():
    workflow = PDExtractionWorkflow()

    request = workflow._request_from_event(
        StartEvent(
            data={
                "content": "Rx Tab Ceftum 500mg BD",
                "content_type": "text",
                "learning_context": {"guidance": "BD means twice daily"},
                "learning_metadata": {"retrieval_matches": 2},
                "extraction_id": "pd_123",
            }
        )
    )

    assert request.content == "Rx Tab Ceftum 500mg BD"
    assert request.content_type == "text"
    assert request.learning_context is not None
    assert request.learning_metadata is not None
    assert request.learning_metadata.retrieval_matches == 2
