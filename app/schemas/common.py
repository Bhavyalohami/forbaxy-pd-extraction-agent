from pydantic import BaseModel


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "ok"


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: str

