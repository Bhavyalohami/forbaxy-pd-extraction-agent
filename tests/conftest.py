import pytest
from app.agents.base import AgentPort, AgentResult
from app.main import create_app
from app.schemas.pd import LearningContext, LearningMetadata
from fastapi.testclient import TestClient


class FakeAgent(AgentPort):
    async def run(
        self,
        session_id: str,
        message: str,
        *,
        learning_context: LearningContext | None = None,
        learning_metadata: LearningMetadata | None = None,
        extraction_id: str | None = None,
    ) -> AgentResult:
        metadata = {"learning_used": learning_context is not None}
        if extraction_id:
            metadata["extraction_id"] = extraction_id
        if learning_metadata and learning_metadata.retrieval_matches is not None:
            metadata["retrieval_matches"] = learning_metadata.retrieval_matches
        return AgentResult(
            response=(
                '{"pd_extraction":{"chief_complaints":["fever"],'
                '"diagnosis":"viral fever",'
                '"extraction_confidence":0.95,'
                '"admission":{"advised":false,"reason":"","ipd_probability":0.3,'
                '"risk_category":"low"}}}'
            ),
            sources=[],
            learning_metadata=metadata,
        )


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        test_client.app.state.container.agent = FakeAgent()
        test_client.app.state.container.chat_service._agent = FakeAgent()
        yield test_client
