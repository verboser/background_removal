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

    if backend == "birefnet":
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        device = settings.device.strip().lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEVICE установлен в cuda, но cuda недоступна")

        torch_device = torch.device(device)
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (settings.birefnet_image_size, settings.birefnet_image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        to_pil = transforms.ToPILImage()
        model = AutoModelForImageSegmentation.from_pretrained(
            settings.birefnet_model_id,
            trust_remote_code=True,
        )
        model.to(torch_device)
        model.float()
        model.eval()

        def predict(image: Image.Image) -> Image.Image:
            size = image.size
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(torch_device)
            with torch.inference_mode():
                alpha = model(tensor)[-1].sigmoid()
            alpha = alpha.detach().float().cpu()[0].squeeze().clamp(0, 1)
            return to_pil(alpha).resize(size, Image.Resampling.LANCZOS).convert("L")

        return {"name": settings.birefnet_model_id, "predict": predict}

    raise ValueError(f"Неподдерживаемый MODEL_BACKEND: {settings.model_backend}")
