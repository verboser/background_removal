from typing import Any

import httpx


class ApiClientError(RuntimeError):
    def __init__(self, detail: str, status_code: int | None = None) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class BackgroundRemovalApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            ) as client:
                response = client.get("/health")
        except httpx.HTTPError as exc:
            raise ApiClientError(
                f"Не удалось подключиться к бэкенду по адресу {self.base_url}: {exc}"
            ) from exc

        if response.is_error:
            try:
                body = response.json()
                detail = body.get("detail")
            except ValueError:
                detail = response.text
            raise ApiClientError(
                detail=str(detail) if detail else f"HTTP ошибка {response.status_code}",
                status_code=response.status_code,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ApiClientError("Бэкенд вернул некорректный ответ health") from exc

    def remove_background(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        files = {"file": (filename, image_bytes, content_type)}
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            ) as client:
                response = client.post("/remove-background", files=files)
        except httpx.HTTPError as exc:
            raise ApiClientError(
                f"Не удалось подключиться к бэкенду по адресу {self.base_url}: {exc}"
            ) from exc

        if response.is_error:
            try:
                body = response.json()
                detail = body.get("detail")
            except ValueError:
                detail = response.text
            raise ApiClientError(
                detail=str(detail) if detail else f"HTTP ошибка {response.status_code}",
                status_code=response.status_code,
            )

        processing_time = response.headers.get("X-Processing-Time-Ms")
        try:
            processing_time_ms = (
                float(processing_time) if processing_time is not None else None
            )
        except ValueError:
            processing_time_ms = None

        return {
            "image_bytes": response.content,
            "processing_time_ms": processing_time_ms,
            "model_backend": response.headers.get("X-Model-Backend"),
            "model_name": response.headers.get("X-Model-Name"),
        }
