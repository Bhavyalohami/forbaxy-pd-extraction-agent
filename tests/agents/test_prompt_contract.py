from app.agents.prompts import PD_SYSTEM_PROMPT


def test_prompt_contains_privacy_boundary_and_json_contract():
    assert "Patient Information" in PD_SYSTEM_PROMPT
    assert "strict JSON only" in PD_SYSTEM_PROMPT
    assert "normal_medicated" in PD_SYSTEM_PROMPT

