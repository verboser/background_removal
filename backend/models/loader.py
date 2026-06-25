from io import BytesIO
import os

from PIL import Image

from backend.config import Settings


def load_background_removal_model(settings: Settings) -> dict:
    backend = settings.model_backend.lower()

    if backend == "rembg":
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError(
                "rembg обязателен для MODEL_BACKEND=rembg. "
                "Установить можно с помощью: pip install 'rembg[cpu]'"
            ) from exc

        session = new_session(settings.rembg_model)

        def predict(image: Image.Image) -> Image.Image:
            result = remove(image.convert("RGB"), session=session)

            if isinstance(result, Image.Image):
                rgba = result.convert("RGBA")
            else:
                rgba = Image.open(BytesIO(result)).convert("RGBA")
                rgba.load()

            alpha = rgba.getchannel("A")
            if alpha.size != image.size:
                alpha = alpha.resize(image.size, Image.Resampling.LANCZOS)
            return alpha

        return {"name": settings.rembg_model, "predict": predict}

        raise ValueError(f"Неподдерживаемый MODEL_BACKEND: {settings.model_backend}")
