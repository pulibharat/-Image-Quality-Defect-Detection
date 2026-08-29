from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Image Quality & Defect Detection API"
    app_version: str = "1.0.0"

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'app.db').as_posix()}"
    model_dir: str = str(BACKEND_DIR / "models_store")
    upload_dir: str = str(BACKEND_DIR / "uploads")

    max_upload_mb: int = 15
    allowed_content_types: str = "image/jpeg,image/png,image/webp,image/bmp,image/tiff"
    cors_origins: str = "*"

    @property
    def allowed_content_types_list(self) -> list[str]:
        return [c.strip() for c in self.allowed_content_types.split(",") if c.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [c.strip() for c in self.cors_origins.split(",") if c.strip()]


settings = Settings()
