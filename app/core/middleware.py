from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(self.header_name, str(uuid4()))
        token = request_id_ctx.set(request_id)
        try:
            logger.info(
                "request_started",
                extra={"method": request.method, "path": request.url.path},
            )
            response = await call_next(request)
            response.headers[self.header_name] = request_id
            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                },
            )
            return response
        finally:
            request_id_ctx.reset(token)
