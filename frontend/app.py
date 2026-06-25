import os
from hashlib import sha256
from io import BytesIO
from PIL import Image, ImageDraw, UnidentifiedImageError
import streamlit as st
from api_client import ApiClientError, BackgroundRemovalApiClient

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

def main() -> None:
    st.set_page_config(page_title="Background Removal", layout="wide")
    st.title("Удаление заднего фона")

    uploaded_file = st.file_uploader("Изображение", type=SUPPORTED_TYPES)
    if uploaded_file is None:
        _clear_current_file_state()
        st.info("Загрузите изображение JPG, PNG или WEBP.")
        return

    image_bytes = uploaded_file.getvalue()
    _reset_result_when_file_changes(uploaded_file.name, image_bytes)

    try:
        input_image = Image.open(BytesIO(image_bytes))
        input_image.load()
    except UnidentifiedImageError:
        st.error("Файл не удалось прочитать как изображение.")
        return

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.subheader("Исходное изображение")
        st.image(input_image, use_container_width=True)
        st.caption(f"{input_image.width}x{input_image.height}px, {uploaded_file.type or 'application/octet-stream'}")

    with right_col:
        st.subheader("Результат")
        result_bytes = st.session_state.get("result_bytes")
        if result_bytes:
            preview = build_checkerboard_preview(result_bytes)
            st.image(preview, use_container_width=True)
            _render_result_metadata(st.session_state.get("result_metadata", {}))

    if st.button("Убрать задний фон", type="primary", use_container_width=True):
        _remove_background(DEFAULT_BACKEND_URL, image_bytes, uploaded_file.name, uploaded_file.type or "application/octet-stream")
        st.rerun()

def build_checkerboard_preview(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    image.load()

    tile = max(10, min(image.size) // 40)
    board = Image.new("RGBA", image.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(board)
    light = (255, 255, 255, 255)
    dark = (238, 238, 238, 255)

    for y in range(0, image.height, tile):
        for x in range(0, image.width, tile):
            color = dark if (x // tile + y // tile) % 2 else light
            draw.rectangle((x, y, x + tile, y + tile), fill=color)

    return Image.alpha_composite(board, image)

def _remove_background(backend_url: str, image_bytes: bytes, filename: str, content_type: str) -> None:
    client = BackgroundRemovalApiClient(backend_url)
    try:
        with st.spinner("Обработка изображения..."):
            result = client.remove_background(image_bytes=image_bytes, filename=filename, content_type=content_type)
    except ApiClientError as exc:
        st.error(exc.detail)
        return

    st.session_state["result_bytes"] = result["image_bytes"]
    st.session_state["result_metadata"] = _result_to_metadata(result)

def _result_to_metadata(result: dict) -> dict:
    metadata = {"Бэкенд": str(result.get("model_backend") or "неизвестно"), "Модель": str(result.get("model_name") or "неизвестно")}
    if "processing_time_ms" in result and result["processing_time_ms"] is not None: metadata["Время"] = f"{result['processing_time_ms']:.0f} ms"
    return metadata

def _render_result_metadata(metadata: dict) -> None:
    if metadata: st.caption(" | ".join(f"{k}: {v}" for k, v in metadata.items()))

def _reset_result_when_file_changes(filename: str, image_bytes: bytes) -> None:
    current_signature = f"{filename}_{sha256(image_bytes).hexdigest()}"
    if st.session_state.get("file_signature") != current_signature:
        st.session_state["file_signature"] = current_signature
        st.session_state.pop("result_bytes", None)
        st.session_state.pop("result_metadata", None)

def _clear_current_file_state() -> None:
    for key in ["file_signature", "result_bytes", "result_metadata"]:
        st.session_state.pop(key, None)

if __name__ == "__main__":
    main()
