def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_chat_creates_session(client):
    response = client.post("/chat", json={"session_id": "abc", "message": "Rx: paracetamol"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "abc"

    session = client.get("/sessions/abc")
    assert session.status_code == 200
    assert len(session.json()["chat_history"]) == 2


def test_delete_missing_session_returns_consistent_error(client):
    response = client.delete("/sessions/missing")
    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "message": "Session 'missing' was not found.",
        "error_code": "NOT_FOUND",
    }


def test_upload_uses_parser_abstraction(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("pd.txt", b"Chief complaint: fever", "text/plain")},
    )
    assert response.status_code == 202
    assert response.json()["parser"] == "mock"
    assert "fever" in response.json()["parsed_text"]


def test_upload_redacts_accidental_pi_like_text(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("pd.txt", b"Patient name: Hidden\nRx: tablet", "text/plain")},
    )
    assert response.status_code == 202
    assert "Patient name" not in response.json()["parsed_text"]
    assert "[PI_REDACTED]" in response.json()["parsed_text"]
