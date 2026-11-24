"""全局配置：通过环境变量 / .env 加载，集中管理所有可调参数。"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql+psycopg://knowledgehub:knowledgehub@postgres:5432/knowledgehub"

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"
    task_queue_enabled: bool = False

    # JWT
    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Embedding
    embedding_provider: str = "fake"  # openai_compatible | fake
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Reranker
    reranker_provider: str = "fallback"  # openai_compatible | none | fallback
    reranker_base_url: str = "https://api.jina.ai/v1"
    reranker_api_key: str = ""
    reranker_model: str = "jina-reranker-v2-base-multilingual"

    # LLM
    llm_provider: str = "fake"  # openai_compatible | fake
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # Retrieval
    vector_top_k: int = 20
    fts_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 10
    final_top_k: int = 5
    min_evidence_score: float = 0.15

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 80

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
