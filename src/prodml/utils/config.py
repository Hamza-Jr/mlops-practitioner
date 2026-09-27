from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from defaults and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------

    project_root: Path = Path(__file__).resolve().parents[3]

    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    models_dir: Path = project_root / "models"
    artifacts_dir: Path = models_dir / "artifacts"

    reports_dir: Path = project_root / "reports"
    evaluation_reports_docs: Path = reports_dir / "model_evaluation.md"
    module_1_report_path: Path = reports_dir / "module-1.md"

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------

    dataset_filename: str = "WildFires_DataSet.csv"
    processed_dataset_filename: str = "WildFires_DataSet_processed.csv"

    target_column: str = "CLASS"

    feature_names: tuple[str, ...] = (
        "NDVI",
        "LST",
        "BURNED_AREA",
    )

    validation_data_path: Path = data_dir / "validation" / "validation_data.csv"

    # Raw dataset labels -> encoded model labels
    class_mapping: dict[str, int] = {
        "no_fire": 0,
        "fire": 1,
    }

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    model_name: str = "baseline"
    model_version: str = "0.1.0"
    training_date: str = "2026-09-19"
    framework: str = "scikit-learn"
    model_candidates: list[str] = [
        "random_forest",
        "logistic_regression",
        "xgboost",
    ]

    random_state: int = 42

    random_forest_params: dict = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    model_path: Path = artifacts_dir / "baseline.pkl"

    onnx_model_path: Path = artifacts_dir / "baseline.onnx"

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    dataset_path: Path = raw_data_dir / dataset_filename

    # ------------------------------------------------------------------
    # # Observed ranges in the training dataset
    # ------------------------------------------------------------------

    training_ranges: dict[str, tuple[float, float]] = {
        "ndvi": (0.030735, 0.781723),
        "lst": (13137.0, 15611.570513),
        "burned_area": (3.0, 9.0),
    }


settings = Settings()
