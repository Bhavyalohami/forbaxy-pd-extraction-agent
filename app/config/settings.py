from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Forbaxy PD Extraction Agent"
    environment: str = "local"
    openai_api_key: str = Field(default="local-dev-key", validation_alias="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_API_BASE",
    )
    model_name: str = Field(default="gpt-4o-mini", validation_alias="MODEL_NAME")
    temperature: float = Field(default=0.1, ge=0, le=2, validation_alias="TEMPERATURE")
    llm_context_window: int = Field(default=128000, validation_alias="LLM_CONTEXT_WINDOW")
    llm_is_function_calling: bool = Field(default=True, validation_alias="LLM_IS_FUNCTION_CALLING")
    agent_timeout_seconds: float = Field(default=30, gt=0, validation_alias="AGENT_TIMEOUT_SECONDS")
    enable_agent_tools: bool = Field(default=True, validation_alias="ENABLE_AGENT_TOOLS")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    request_id_header: str = Field(default="X-Request-ID", validation_alias="REQUEST_ID_HEADER")
    max_upload_size_mb: int = Field(default=20, validation_alias="MAX_UPLOAD_SIZE_MB")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
