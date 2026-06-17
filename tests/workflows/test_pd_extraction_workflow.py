from app.workflows.pd_extraction import PDExtractionWorkflow
from llama_index.core.workflow import StartEvent


def test_message_from_event_prefers_message_key():
    workflow = PDExtractionWorkflow()

    assert workflow._message_from_event(StartEvent(message="Rx text")) == "Rx text"
