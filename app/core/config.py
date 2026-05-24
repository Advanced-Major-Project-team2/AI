# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

from pathlib import Path


class Settings(BaseSettings):
    # --- 앱 기본 ---
    APP_NAME: str = Field(default="gangku-ai-server")
    ENV: str = Field(default="local")
    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)

    # --- CORS/로깅 ---
    CORS_ORIGINS: List[str] = Field(default_factory=list)
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # .env에서 소문자/대문자 모두 허용
        extra="ignore",
    )

    # --- RECOMMANDATIONS ----
    RECOMMANDS_LIMIT: int = Field(default=50)
    RECOMMENDER_N_CLUSTERS: int = Field(default=5)
    RECOMMENDER_SVD_DIM: int = Field(default=4)

    # 아티팩트 DIRECTORY
    CLUSTER_ARTIFACT_DIR: Path = Path("app/cluster/artifacts/user_clustering")
    POPULARITY_ARTIFACT_DIR: Path = Path("app/cluster/artifacts/popularity")


settings = Settings()
