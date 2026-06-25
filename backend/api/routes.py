from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response

from backend.config import Settings
from backend.api.image_io import image_to_png_bytes, read_upload_image, compose_rgba


router = APIRouter()


def get_context(request: Request) -> dict[str, Any]:
    return request.app.extra["context"]


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    context = get_context(request)
    settings: Settings = context["settings"]
    model = context["model"]
    return {
        "status": "ok",
        "model_backend": settings.model_backend,
        "model_name": model["name"],
    }


@router.post("/remove-background")
def remove_background(
    file: UploadFile = File(...),
    context: dict[str, Any] = Depends(get_context),
) -> Response:
    started_at = perf_counter()
    settings: Settings = context["settings"]
    model = context["model"]
    image = read_upload_image(file, settings)
    alpha = model["predict"](image)
    result = compose_rgba(image, alpha)
    payload = image_to_png_bytes(result)
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    return Response(
        content=payload,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="background_removed.png"',
            "X-Model-Backend": settings.model_backend,
            "X-Model-Name": model["name"],
            "X-Processing-Time-Ms": str(elapsed_ms),
        },
    )
