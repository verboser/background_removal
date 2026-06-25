from fastapi import FastAPI

from backend.api.routes import router
from backend.config import get_settings
from backend.models.loader import load_background_removal_model


def create_app() -> FastAPI:
    settings = get_settings()
    context = {
        "settings": settings,
        "model": load_background_removal_model(settings),
    }
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        context=context,
    )
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
