from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from app.core.logging import get_logger
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "APP_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, message: str, *, error_code: str = "NOT_FOUND") -> None:
        super().__init__(message, error_code=error_code, status_code=status.HTTP_404_NOT_FOUND)


class PrivacyBoundaryError(AppError):
    def __init__(self, message: str = "Patient information is not accepted by this agent.") -> None:
        super().__init__(
            message,
            error_code="PRIVACY_BOUNDARY",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


def _error_payload(message: str, error_code: str) -> dict[str, object]:
    return ErrorResponse(message=message, error_code=error_code).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("application_error", extra={"error_code": exc.error_code})
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.message, exc.error_code),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("validation_error", extra={"errors": exc.errors()})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload("Request validation failed.", "VALIDATION_ERROR"),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"exception_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("An unexpected error occurred.", "INTERNAL_ERROR"),
        )
