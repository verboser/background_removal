import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import streamlit as st
from api_client import ApiClientError, BackgroundRemovalApiClient

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

def main() -> None:
    st.set_page_config(page_title="Background Removal", layout="wide")
    st.title("Удаление заднего фона")

    uploaded_file = st.file_uploader("Изображение", type=SUPPORTED_TYPES)
    if uploaded_file is None:
        st.info("Загрузите изображение JPG, PNG или WEBP.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        input_image = Image.open(BytesIO(image_bytes))
        input_image.load()
    except UnidentifiedImageError:
        st.error("Файл не удалось прочитать как изображение.")
        return

    st.subheader("Исходное изображение")
    st.image(input_image, use_container_width=True)

    if st.button("Убрать задний фон", type="primary", use_container_width=True):
        client = BackgroundRemovalApiClient(DEFAULT_BACKEND_URL)
        try:
            with st.spinner("Обработка изображения..."):
                result = client.remove_background(image_bytes=image_bytes, filename=uploaded_file.name, content_type=uploaded_file.type or "application/octet-stream")
            st.subheader("Результат")
            st.image(result["image_bytes"], use_container_width=True)
            st.success("Готово!")
        except ApiClientError as exc:
            st.error(exc.detail)

if __name__ == "__main__":
    main()
