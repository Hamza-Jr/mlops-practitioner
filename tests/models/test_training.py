from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from prodml.models.evaluate import evaluate_model
from prodml.models.factory import create_model
from prodml.models.train import save_model, train_model


def test_create_model_returns_random_forest() -> None:
    model = create_model("random_forest")

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 42


def test_create_model_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model("unsupported_model")


def test_train_model_fits_model() -> None:
    model = create_model("random_forest")

    X_train = pd.DataFrame(
        {
            "NDVI": [0.1, 0.8, 0.2, 0.7],
            "LST": [318.0, 295.0, 315.0, 298.0],
            "BURNED_AREA": [10.0, 0.0, 8.0, 0.0],
        }
    )
    y_train = pd.Series([1, 0, 1, 0])

    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    predictions = trained_model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_evaluate_model_returns_metrics_and_writes_report(
    tmp_path: Path,
) -> None:
    model = create_model("random_forest")

    X_train = pd.DataFrame(
        {
            "NDVI": [0.1, 0.8, 0.2, 0.7],
            "LST": [318.0, 295.0, 315.0, 298.0],
            "BURNED_AREA": [10.0, 0.0, 8.0, 0.0],
        }
    )
    y_train = pd.Series([1, 0, 1, 0])

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    X_test = pd.DataFrame(
        {
            "NDVI": [0.15, 0.75, 0.25, 0.65],
            "LST": [316.0, 296.0, 314.0, 299.0],
            "BURNED_AREA": [7.0, 0.0, 5.0, 0.0],
        }
    )
    y_test = pd.Series([1, 0, 1, 0])

    report_path = tmp_path / "evaluation.md"

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="random_forest",
        report_path=report_path,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], list)

    assert report_path.exists()
    assert "Model Evaluation" in report_path.read_text(encoding="utf-8")


def test_save_model_creates_artifact(tmp_path: Path) -> None:
    model = create_model("random_forest")

    X_train = pd.DataFrame(
        {
            "NDVI": [0.1, 0.8, 0.2, 0.7],
            "LST": [318.0, 295.0, 315.0, 298.0],
            "BURNED_AREA": [10.0, 0.0, 8.0, 0.0],
        }
    )
    y_train = pd.Series([1, 0, 1, 0])

    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    artifact_path = tmp_path / "artifacts" / "model.pkl"

    save_model(
        model=trained_model,
        model_path=artifact_path,
    )

    assert artifact_path.exists()
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size > 0


def test_evaluate_model_appends_to_existing_report(
    tmp_path: Path,
) -> None:
    """Verify evaluation appends to an existing report."""

    model = create_model("random_forest")

    X_train = pd.DataFrame(
        {
            "NDVI": [0.1, 0.8, 0.2, 0.7],
            "LST": [318.0, 295.0, 315.0, 298.0],
            "BURNED_AREA": [10.0, 0.0, 8.0, 0.0],
        }
    )
    y_train = pd.Series([1, 0, 1, 0])

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    X_test = pd.DataFrame(
        {
            "NDVI": [0.15, 0.75],
            "LST": [316.0, 296.0],
            "BURNED_AREA": [7.0, 0.0],
        }
    )
    y_test = pd.Series([1, 0])

    report_path = tmp_path / "evaluation.md"

    # Simulate a report that already exists.
    report_path.write_text(
        "Previous evaluation\n",
        encoding="utf-8",
    )

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="random_forest",
        report_path=report_path,
    )

    report = report_path.read_text(encoding="utf-8")

    # Existing content must not be removed.
    assert "Previous evaluation" in report

    # New evaluation must be appended.
    assert "# Model Evaluation" in report
    assert "**Model:** random_forest" in report
    assert "**Model Type:** RandomForestClassifier" in report
    assert "Accuracy:" in report
    assert "F1-Score:" in report
    assert "Confusion Matrix" in report

    # The separator confirms the append branch was used.
    assert "---" in report
