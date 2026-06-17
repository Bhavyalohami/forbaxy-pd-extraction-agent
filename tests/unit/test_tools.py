import pytest
from app.tools.builtin import calculator, prescription_validator_placeholder
from app.tools.registry import build_default_registry


@pytest.mark.asyncio
async def test_calculator_allows_safe_arithmetic():
    assert await calculator("2 + 3 * 4") == "14.0"


@pytest.mark.asyncio
async def test_validator_rejects_pi_like_context():
    result = await prescription_validator_placeholder('{"patient_name":"x"}')
    assert "Rejected" in result


def test_registry_exports_llamaindex_tools():
    tools = build_default_registry().as_llamaindex_tools()
    assert {tool.metadata.name for tool in tools} >= {"current_datetime", "calculator"}

