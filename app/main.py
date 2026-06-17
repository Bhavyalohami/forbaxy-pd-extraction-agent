from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import Settings
from app.core.container import Container
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIDMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_logging(settings.log_level)
    app.state.container = Container.build(settings)
    yield


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="PD-only prescription extraction and review-learning agent.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIDMiddleware, header_name=settings.request_id_header)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()

