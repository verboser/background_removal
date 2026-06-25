from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.config import Settings


UPLOAD_READ_CHUNK_SIZE = 1024 * 1024

SUPPORTED_CONTENT_TYPES = {
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def read_upload_image(file: UploadFile, settings: Settings) -> Image.Image:
    if file.content_type and file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Неподдерживаемый тип файла: {file.content_type}",
        )

    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = file.file.read(UPLOAD_READ_CHUNK_SIZE)
        if not chunk:
            break

        total_size += len(chunk)
        chunks.append(chunk)

    data = b"".join(chunks)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загруженный файл пуст",
        )

    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            pixel_count = width * height

            #             image.load()
            return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Слишком большая картинка",
        ) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не подходящая картинка",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не подходящая картинка",
        ) from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def compose_rgba(image: Image.Image, alpha: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    rgba = rgb.convert("RGBA")
    alpha = alpha.convert("L")
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, Image.Resampling.LANCZOS)
    rgba.putalpha(alpha)
    return rgba
