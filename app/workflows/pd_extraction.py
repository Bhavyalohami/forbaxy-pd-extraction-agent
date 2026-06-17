from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAIError

from app.agents.prescription_agent import PrescriptionAgent
from app.agents.prompts import PD_SYSTEM_PROMPT
from app.config.settings import Settings
from app.core.exceptions import AppError
from app.schemas.prescription import PDExtractionResponse
from app.services.privacy import redact_patient_information


class PDExtractionWorkflow(Workflow):
    """Deployable LlamaAgents workflow for cropped PD-only extraction."""

    @step()
    async def extract(self, ev: StartEvent) -> StopEvent:
        message = self._message_from_event(ev)
        settings = Settings()
        llm = OpenAILike(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.temperature,
            context_window=settings.llm_context_window,
            is_chat_model=True,
            is_function_calling_model=settings.llm_is_function_calling,
        )

        prompt = (
            f"{PD_SYSTEM_PROMPT}\n\n"
            "Context: Input is expected to contain only cropped Prescription Details. "
            "Return only the required JSON object.\n\n"
            f"Task input:\n{redact_patient_information(message)}"
        )

        try:
            response = await llm.acomplete(prompt)
        except OpenAIError as exc:
            raise AppError(
                "LLM provider request failed.",
                error_code="LLM_PROVIDER_ERROR",
                status_code=502,
            ) from exc

        json_text = PrescriptionAgent.extract_json_object(
            redact_patient_information(str(response))
        )
        extraction = PDExtractionResponse.model_validate_json(json_text)
        return StopEvent(result=extraction.model_dump())

    def _message_from_event(self, ev: StartEvent) -> str:
        for key in ("message", "query", "input", "text"):
            value = ev.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(ev.get("data", ""))


pd_extraction_workflow = PDExtractionWorkflow(timeout=45)
