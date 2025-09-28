"""
Configuration settings for the Knowledge Distillation Ensemble project.
Uses Pydantic v2 BaseSettings for environment variable management.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "knowledge_distillation_ensemble"
ML_ROOT = SRC_ROOT / "ml"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    The model_config allows extra fields to be ignored, preventing errors
    when the .env file contains additional key-value pairs not defined here.
    """

    # Model configuration for Pydantic v2
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # This prevents errors from extra fields in .env
        case_sensitive=False,
    )

    # Environment variables
    KAGGLE_USERNAME: str = None
    KAGGLE_KEY: str = None
    KAGGLEHUB_CACHE: str = "~/.cache/kagglehub/"
    KAGGLE_LOGGING_ROOT_DIR: str = PROJECT_ROOT.as_posix()

    # Application settings
    app_name: str = "Knowledge Distillation Ensemble"
    debug: bool = False

    # ML/Training related settings
    random_seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100

    # Data settings
    data_path: Path = PROJECT_ROOT / "data"
    model_save_path: Path = ML_ROOT / "models"

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Optional database settings (if needed)
    database_url: Optional[str] = None

    # Optional API settings (if needed)
    api_host: str = "localhost"
    api_port: int = 8000


# Create a global settings instance
settings = Settings()
