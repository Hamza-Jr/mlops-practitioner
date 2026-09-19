from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from defaults and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    models_dir: Path = project_root / "models"
    artifacts_dir: Path = models_dir / "artifacts"
    model_path: Path = artifacts_dir / "baseline.pkl"

    # Model configuration
    model_name: str = "baseline"
    model_version: str = "0.1.0"

    # Inference features
    feature_names: tuple[str, ...] = (
        "NDVI",
        "LST",
        "BURNED_AREA",
    )

    # Class mapping
    class_mapping: dict[int, str] = {
        0: "no_fire",
        1: "fire",
    }


settings = Settings()
