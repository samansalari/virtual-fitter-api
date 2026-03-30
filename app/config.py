from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    environment: str = Field(default="production", validation_alias="ENVIRONMENT")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    storage_dir: Path = Field(default=Path("./storage"), validation_alias="VF_STORAGE_DIR")
    media_base_url: str = Field(default="http://localhost:8000/media", validation_alias="VF_MEDIA_BASE_URL")
    render_mode: str = Field(default="overlay", validation_alias="VF_RENDER_MODE")
    api_token: str = Field(default="", validation_alias="VF_API_TOKEN")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "https://kits.style",
            "https://www.kits.style",
            "https://kits-uk.myshopify.com",
            "http://127.0.0.1:9292",
            "http://localhost:9292",
        ],
        validation_alias="CORS_ORIGINS",
    )
    shopify_admin_access_token: str = Field(default="", validation_alias="SHOPIFY_ADMIN_ACCESS_TOKEN")
    shopify_api_version: str = Field(default="2025-10", validation_alias="SHOPIFY_API_VERSION")
    shopify_default_shop_domain: str = Field(default="", validation_alias="SHOPIFY_SHOP_DOMAIN")
    request_timeout_seconds: float = Field(default=20.0, validation_alias="VF_REQUEST_TIMEOUT_SECONDS")
    segmentation_confidence_threshold: float = Field(
        default=0.70,
        validation_alias=AliasChoices("SEGMENTATION_MIN_CONFIDENCE", "VF_SEGMENTATION_CONFIDENCE"),
    )
    low_confidence_threshold: float = Field(
        default=0.45,
        validation_alias=AliasChoices("OPENCV_CONFIDENCE_THRESHOLD", "VF_LOW_CONFIDENCE_THRESHOLD"),
    )
    replicate_api_token: str = Field(default="", validation_alias="REPLICATE_API_TOKEN")
    segmentation_prefer_sam2: bool = Field(default=True, validation_alias="SEGMENTATION_PREFER_SAM2")
    segmentation_sam2_timeout: int = Field(default=120, validation_alias="SEGMENTATION_SAM2_TIMEOUT")
    enable_sam2: bool = Field(default=True, validation_alias="ENABLE_SAM2")
    enable_opencv_fallback: bool = Field(default=True, validation_alias="ENABLE_OPENCV_FALLBACK")
    segmentation_sam2_model: str = Field(default="meta/sam-2-base", validation_alias="SEGMENTATION_SAM2_MODEL")
    segmentation_sam2_fallback_model: str = Field(default="meta/sam-2-large", validation_alias="SEGMENTATION_SAM2_FALLBACK_MODEL")
    enable_ai_rendering: bool = Field(default=True, validation_alias="ENABLE_AI_RENDERING")
    render_ai_timeout: int = Field(default=180, validation_alias="RENDER_AI_TIMEOUT")
    flux_ip_adapter_model: str = Field(default="lucataco/flux-dev-ip-adapter", validation_alias="FLUX_IP_ADAPTER_MODEL")
    flux_inpaint_model: str = Field(default="black-forest-labs/flux-fill-dev", validation_alias="FLUX_INPAINT_MODEL")
    flux_ip_adapter_scale: float = Field(default=0.75, validation_alias="FLUX_IP_ADAPTER_SCALE")
    flux_guidance_scale: float = Field(default=7.5, validation_alias="FLUX_GUIDANCE_SCALE")
    flux_num_inference_steps: int = Field(default=28, validation_alias="FLUX_NUM_INFERENCE_STEPS")
    flux_strength: float = Field(default=0.85, validation_alias="FLUX_STRENGTH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"1", "true", "yes", "on", "debug", "development"}:
                return True
            if stripped in {"0", "false", "no", "off", "release", "production"}:
                return False
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir = settings.storage_dir.resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.media_base_url = settings.media_base_url.rstrip("/")
    return settings
