from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "Background Removal Service"
    api_prefix: str = ""

    model_backend: str = "rembg"
    rembg_model: str = "birefnet-general"
    birefnet_model_id: str = "ZhengPeng7/BiRefNet"
    birefnet_image_size: int = Field(default=512, ge=256, le=2048)
    device: str = "cpu"

    max_image_size_mb: int = Field(default=10, ge=1, le=100)
    max_image_pixels: int = Field(default=12_000_000, ge=1_000_000)

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
