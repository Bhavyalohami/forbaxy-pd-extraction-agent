from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    parsed_text: str
    parser: str
    metadata: dict[str, str] = Field(default_factory=dict)

