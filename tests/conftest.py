import pytest
from app.agents.base import AgentPort, AgentResult
from app.main import create_app
from fastapi.testclient import TestClient


class FakeAgent(AgentPort):
    async def run(self, session_id: str, message: str) -> AgentResult:
        return AgentResult(response='{"pd_extraction":{"chief_complaints":[]}}', sources=[])


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        test_client.app.state.container.agent = FakeAgent()
        test_client.app.state.container.chat_service._agent = FakeAgent()
        yield test_client

