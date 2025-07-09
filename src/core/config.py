"""Configuration settings for the Chat with PDF application."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    # LLM Settings
    model_name: str = Field(default="gemini-2.5-flash", env="MODEL_NAME")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")

    # Vector Store Settings
    vector_store_path: str = Field(
        default="./data/vectorstore", env="VECTOR_STORE_PATH")
    pdf_storage_path: str = Field(
        default="./data/pdfs", env="PDF_STORAGE_PATH")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    embedding_model: str = Field(
        default="models/text-embedding-004", env="EMBEDDING_MODEL")

    # API Settings
    max_memory_messages: int = Field(default=10, env="MAX_MEMORY_MESSAGES")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Router Agent Settings
    router_confidence_threshold: float = Field(
        default=0.7, env="ROUTER_CONFIDENCE_THRESHOLD")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(settings.vector_store_path, exist_ok=True)
    os.makedirs(settings.pdf_storage_path, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
