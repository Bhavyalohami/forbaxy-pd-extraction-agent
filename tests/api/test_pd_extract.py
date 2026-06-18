def test_pd_extract_returns_production_response(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "Chief complaint fever. Diagnosis viral fever.",
            "content_type": "text",
            "extraction_id": "pd_123",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence_score"] == 95
    assert payload["ipd_probability"] == 30
    assert payload["patient"]["diagnosis"] == "viral fever"
    assert payload["learning_metadata"] == {
        "learning_used": False,
        "extraction_id": "pd_123",
    }


def test_pd_extract_accepts_learning_context_and_metadata(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "Rx Paracetamol 500mg BD.",
            "content_type": "text",
            "learning_context": {
                "guidance": "Prefer visible dosage evidence.",
                "patterns": ["HTN means hypertension"],
                "common_corrections": ["Dosage is often omitted"],
            },
            "learning_metadata": {"retrieval_matches": 3, "average_similarity": 0.89},
        },
    )

    assert response.status_code == 200
    metadata = response.json()["learning_metadata"]
    assert metadata["learning_used"] is True
    assert metadata["retrieval_matches"] == 3


def test_pd_extract_rejects_patient_information_content(client):
    response = client.post(
        "/pd/extract",
        json={"content": "Patient name: Hidden\nRx tablet", "content_type": "text"},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "PRIVACY_BOUNDARY"


def test_pd_extract_forbids_patient_information_fields(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "Rx tablet",
            "content_type": "text",
            "patient_name": "Hidden",
        },
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
