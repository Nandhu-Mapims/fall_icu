# Purpose: FastAPI application entrypoint for ICU patient fall detection service.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers.monitoring import router as monitoring_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(monitoring_router, prefix=settings.api_prefix, tags=["monitoring"])
    return app


app = create_app()
