"""
Configuration management using Pydantic Settings.
Loads settings from environment variables or .env file with robust typing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Info
    PROJECT_NAME: str = "Multimodal Enterprise RAG"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIRECTORY: str = "./data/uploads"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"

    # Ingestion & Chunking
    CHUNK_SIZE_TOKENS: int = 800
    CHUNK_OVERLAP_TOKENS: int = 150
    ENABLE_OCR_FALLBACK: bool = True
    OCR_MIN_TEXT_THRESHOLD: int = 50

    # Embeddings
    EMBEDDING_PROVIDER: Literal["huggingface", "openai"] = "huggingface"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Retrieval & Search Hyperparameters
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    RRF_K: int = 60
    RERANKER_TOP_K: int = 5
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM Settings
    LLM_PROVIDER: Literal["openai", "groq", "ollama"] = "openai"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1500
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIRECTORY)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        path = Path(self.CHROMA_PERSIST_DIRECTORY)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
