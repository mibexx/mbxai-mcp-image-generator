import importlib.metadata
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent
SERVICE_NAME = "OPENAI_IMAGE_GENERATOR_"


def _get_version() -> str:
    """Get the package version."""
    try:
        return importlib.metadata.version("openai_image_generator")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"  # Default during development


class ApplicationConfig(BaseSettings):
    """Application configuration."""

    name: str = "OpenAI Image generator"
    description: str = "AI tool to create images from prompt using OpenAI's image generation API"
    version: str = Field(default_factory=_get_version)
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(
        env_prefix=SERVICE_NAME,
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OpenRouterAPIConfig(BaseSettings):
    """OpenRouter API configuration."""

    api_key: str = Field(alias="OPENROUTER_TOKEN")
    base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(alias="OPENAI_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    image_model: str = Field(default="gpt-image-1", alias="OPENAI_IMAGE_MODEL")

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
    
    def _validate_config(self):
        """Validate the OpenAI configuration."""
        logger = logging.getLogger(__name__)
        
        logger.info("=== OpenAI Configuration Validation ===")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Image Model: {self.image_model}")
        logger.info(f"API Key present: {'Yes' if self.api_key else 'No'}")
        
        # Validate API key format (basic check)
        if not self.api_key:
            logger.error("OpenAI API key is missing!")
        elif not self.api_key.startswith('sk-'):
            logger.warning("OpenAI API key doesn't start with 'sk-' - this might be invalid")
        
        # Validate base URL
        if not self.base_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid base URL format: {self.base_url}")
        
        # Validate image model
        valid_models = ['dall-e-2', 'dall-e-3', 'gpt-image-1']
        if self.image_model not in valid_models:
            logger.warning(f"Image model '{self.image_model}' is not in the standard list: {valid_models}")
        
        logger.info("Configuration validation completed")


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    server_url: str | None = Field(default=None, alias="MCP_SERVER_URL")

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

class ServiceAPIConfig(BaseSettings):
    """Service API configuration."""

    api_url: str = Field(default="https://api.mbxai.cloud/api", alias="MBXAI_API_URL")
    token: str = Field(default="", alias="MBXAI_API_TOKEN")
    service_namespace: str = Field(default="mbxai-srv", alias="SERVICE_NAMESPACE")

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ImageStorageConfig(BaseSettings):
    """Image storage and proxy configuration."""

    storage_path: str = Field(default="/app/uploads", alias="IMAGE_STORAGE_PATH")
    proxy_domain: str = Field(default="http://image-generator.mbxai-mvc.svc.cluster.local:5000", alias="IMAGE_PROXY_DOMAIN")
    proxy_endpoint: str = Field(default="/images", alias="IMAGE_PROXY_ENDPOINT")

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_config() -> ApplicationConfig:
    """Get the application configuration singleton."""
    return ApplicationConfig()


@lru_cache
def get_openrouter_api_config() -> OpenRouterAPIConfig:
    """Get the OpenRouter API configuration singleton."""
    return OpenRouterAPIConfig()


@lru_cache
def get_openai_config() -> OpenAIConfig:
    """Get the OpenAI API configuration singleton."""
    return OpenAIConfig()


@lru_cache
def get_service_api_config() -> ServiceAPIConfig:
    """Get the service api configuration singleton."""
    return ServiceAPIConfig()


@lru_cache
def get_image_storage_config() -> ImageStorageConfig:
    """Get the image storage configuration singleton."""
    return ImageStorageConfig()