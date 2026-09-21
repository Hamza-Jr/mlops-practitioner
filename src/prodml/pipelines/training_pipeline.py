from pathlib import Path

from sklearn.model_selection import train_test_split

from prodml.models.evaluate import evaluate_model
from prodml.models.factory import create_model
from prodml.models.train import save_model, train_model
from prodml.pipelines.data_pipeline import run_data_pipeline
from prodml.utils.config import settings


def run_training_pipeline(
    model_name: str,
    model_path: Path | None = None,
) -> dict[str, float | list[list[int]]]:
    """Run the complete model training pipeline."""

    # ===============================================================
    # 1. Data pipeline
    # ===============================================================

    X, y = run_data_pipeline()

    # ===============================================================
    # 2. Train/test split
    # ===============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_state,
        stratify=y,
    )

    # ===============================================================
    # 3. Create model
    # ===============================================================

    model = create_model(model_name)

    # ===============================================================
    # 4. Train model
    # ===============================================================

    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    # ===============================================================
    # 5. Evaluate model
    # ===============================================================

    metrics = evaluate_model(
        model=trained_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        report_path=settings.evaluation_reports_docs,
    )

    # ===============================================================
    # 6. Persist trained model
    # ===============================================================

    artifact_path = model_path or settings.artifacts_dir / f"{model_name}.pkl"

    save_model(
        model=trained_model,
        model_path=artifact_path,
    )

    return metrics
