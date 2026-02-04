"""
MINDYARD - Core Configuration
システム全体の設定を管理
"""
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# デフォルトのLLM設定
DEFAULT_LLM_CONFIG_FAST = {"provider": "openai", "model": "gpt-4o-mini"}
DEFAULT_LLM_CONFIG_BALANCED = {"provider": "openai", "model": "gpt-4o"}
DEFAULT_LLM_CONFIG_DEEP = {"provider": "openai", "model": "o1-preview"}


class Settings(BaseSettings):
    """アプリケーション設定"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "MINDYARD"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_v1_prefix: str = "/api/v1"

    # Security
    secret_key: str = Field(default="change-me-in-production-mindyard-secret-key-2024")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://mindyard:mindyard@localhost:5432/mindyard"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "mindyard_insights"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-small"

    # Vertex AI (Google Cloud)
    vertex_project_id: Optional[str] = None
    vertex_location: str = "us-central1"

    # LLM Configuration (Multi-Provider, JSON format)
    # 環境変数でJSON文字列として設定可能
    # 例: LLM_CONFIG_FAST='{"provider": "openai", "model": "gpt-4o-mini"}'
    # 例: LLM_CONFIG_BALANCED='{"provider": "vertex", "model": "gemini-1.5-flash"}'
    llm_config_fast: str = Field(
        default=json.dumps(DEFAULT_LLM_CONFIG_FAST),
        description="FAST tier LLM config (JSON string)"
    )
    llm_config_balanced: str = Field(
        default=json.dumps(DEFAULT_LLM_CONFIG_BALANCED),
        description="BALANCED tier LLM config (JSON string)"
    )
    llm_config_deep: str = Field(
        default=json.dumps(DEFAULT_LLM_CONFIG_DEEP),
        description="DEEP tier LLM config (JSON string)"
    )

    # Legacy LLM Model Configuration (deprecated, use llm_config_* instead)
    # 後方互換性のために残す
    llm_model_deep: str = "gpt-5.2"
    llm_model_balanced: str = "gpt-5-mini"
    llm_model_fast: str = "gpt-5-nano"
    openai_model: str = "gpt-4-turbo-preview"

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # Layer 2 Processing
    sharing_threshold_score: int = 70  # 共有価値スコアの閾値

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Logging
    log_level: str = "INFO"

    def get_llm_config(self, role: str) -> Dict[str, Any]:
        """
        用途別のLLM設定を取得

        Args:
            role: "fast", "balanced", or "deep"

        Returns:
            {"provider": "openai"|"vertex", "model": "model-name", ...}
        """
        config_map = {
            "fast": self.llm_config_fast,
            "balanced": self.llm_config_balanced,
            "deep": self.llm_config_deep,
        }

        config_str = config_map.get(role, self.llm_config_balanced)

        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            # パースに失敗した場合はデフォルト設定を返す
            defaults = {
                "fast": DEFAULT_LLM_CONFIG_FAST,
                "balanced": DEFAULT_LLM_CONFIG_BALANCED,
                "deep": DEFAULT_LLM_CONFIG_DEEP,
            }
            config = defaults.get(role, DEFAULT_LLM_CONFIG_BALANCED)

        return config

    def is_openai_available(self) -> bool:
        """OpenAI APIが利用可能かどうか"""
        return bool(self.openai_api_key)

    def is_vertex_available(self) -> bool:
        """Vertex AIが利用可能かどうか（ADCまたはプロジェクトID）"""
        # プロジェクトIDが設定されているか、ADCが設定されていれば利用可能
        # 実際の認証チェックはプロバイダー初期化時に行う
        return True  # ADCは環境によって自動検出されるため常にtrueを返す


@lru_cache
def get_settings() -> Settings:
    """設定のシングルトンインスタンスを取得"""
    return Settings()


settings = get_settings()
