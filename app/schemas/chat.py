from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    response: str
    sources: list[dict[str, str]] = Field(default_factory=list)
    session_id: str

