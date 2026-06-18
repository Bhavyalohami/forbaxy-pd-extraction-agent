from app.agents.prescription_agent import PrescriptionAgent


def test_extract_json_object_from_fenced_response():
    response = 'Here is the result:\n```json\n{"pd_extraction": {}}\n```'

    assert PrescriptionAgent.extract_json_object(response) == '{"pd_extraction": {}}'


def test_extract_json_object_from_wrapped_response():
    response = 'prefix {"pd_extraction": {"notes": ""}} suffix'

    assert PrescriptionAgent.extract_json_object(response) == '{"pd_extraction": {"notes": ""}}'


def test_coerce_normalises_string_investigations():
    agent = object.__new__(PrescriptionAgent)
    response = agent._coerce_pd_json(
        """
        ```json
        {
          "pd_extraction": {
            "investigations": ["USG Rt buttock to rule out abscess"],
            "medicines": [{"name": "Ceftum", "confidence": 80}],
            "admission": {"ipd_probability": 30},
            "medication_assessment": {"status": "unclear", "confidence": 70},
            "extraction_confidence": 70
          }
        }
        ```
        """
    )

    assert '"name":"USG Rt buttock to rule out abscess"' in response
    assert '"confidence":0.8' in response
    assert '"ipd_probability":0.3' in response
    assert '"extraction_confidence":0.7' in response
