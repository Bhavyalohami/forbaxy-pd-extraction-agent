from app.agents.prescription_agent import PrescriptionAgent


def test_extract_json_object_from_fenced_response():
    response = 'Here is the result:\n```json\n{"pd_extraction": {}}\n```'

    assert PrescriptionAgent.extract_json_object(response) == '{"pd_extraction": {}}'


def test_extract_json_object_from_wrapped_response():
    response = 'prefix {"pd_extraction": {"notes": ""}} suffix'

    assert PrescriptionAgent.extract_json_object(response) == '{"pd_extraction": {"notes": ""}}'
