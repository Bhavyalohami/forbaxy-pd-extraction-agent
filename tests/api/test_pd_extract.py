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
    assert payload["admission_advised"] is False
    assert payload["follow_up_date"] == "2026-06-21"
    assert payload["consultant"] == {
        "name": "Dr Sample",
        "department": "General Medicine",
        "specialty": "Physician",
    }
    assert payload["patient"]["diagnosis"] == "viral fever"
    assert payload["patient"]["follow_up"] == "review after 3 days"
    assert payload["vitals"]["rbs"] == ""
    assert payload["issues"] == ["dose"]
    assert payload["learning_metadata"] == {
        "learning_used": False,
        "retrieval_matches": 0,
        "average_similarity": None,
        "context_size": 0,
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


def test_pd_extract_does_not_treat_empty_learning_context_as_used(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "Rx Paracetamol 500mg BD.",
            "content_type": "text",
            "learning_context": {"guidance": "", "patterns": [], "common_corrections": []},
            "learning_metadata": {"learning_used": True, "retrieval_matches": 3},
        },
    )

    assert response.status_code == 200
    metadata = response.json()["learning_metadata"]
    assert metadata["learning_used"] is False
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


def test_pd_extract_parses_raw_image_payloads(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "data:image/jpeg;base64," + ("QUJD" * 80),
            "content_type": "image",
        },
    )

    assert response.status_code == 200


def test_pd_extract_accepts_image_label_with_extracted_text(client):
    response = client.post(
        "/pd/extract",
        json={
            "content": "Chief complaint fever. Rx Paracetamol 500mg BD.",
            "content_type": "image",
        },
    )

    assert response.status_code == 200
