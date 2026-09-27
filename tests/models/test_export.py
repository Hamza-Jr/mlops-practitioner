from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from prodml.models.export import export_model
from prodml.utils.config import settings


def test_export_model_raises_when_input_does_not_exist(
    tmp_path: Path,
) -> None:
    """Export fails clearly when the trained model artifact is missing."""

    missing_model = tmp_path / "missing_model.pkl"
    output_path = tmp_path / "model.onnx"

    try:
        export_model(
            input_path=missing_model,
            output_path=output_path,
        )
    except FileNotFoundError as exc:
        assert "Model artifact not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_export_model_creates_onnx_artifact(
    tmp_path: Path,
) -> None:
    """A trained sklearn model can be exported to an ONNX artifact."""

    model_path = tmp_path / "model.pkl"
    output_path = tmp_path / "artifacts" / "model.onnx"

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=settings.random_state,
    )

    X = [
        [0.1, 310.0, 5.0],
        [0.8, 295.0, 0.0],
        [0.2, 315.0, 8.0],
        [0.9, 290.0, 0.0],
    ]
    y = [1, 0, 1, 0]

    model.fit(X, y)
    joblib.dump(model, model_path)

    result = export_model(
        input_path=model_path,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
