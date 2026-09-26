from pathlib import Path

from prodml.pipelines.training_pipeline import run_training_pipeline
from prodml.utils.config import settings


def test_run_training_pipeline_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the complete training pipeline produces metrics and a model artifact."""

    dataset = tmp_path / "training_data.csv"

    dataset.write_text(
        """NDVI;LST;BURNED_AREA;CLASS
0.12;318.5;12.5;fire
0.45;305.2;0.0;no_fire
0.78;298.1;0.0;no_fire
0.33;310.0;5.2;fire
0.89;292.0;0.0;no_fire
0.22;315.0;8.0;fire
0.15;320.0;15.0;fire
0.67;300.0;0.0;no_fire
0.28;316.0;10.0;fire
0.82;295.0;0.0;no_fire
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "dataset_path", dataset)
    monkeypatch.setattr(
        settings,
        "processed_data_dir",
        tmp_path / "processed",
    )
    monkeypatch.setattr(
        settings,
        "evaluation_reports_docs",
        tmp_path / "model_evaluation.md",
    )

    artifact_path = tmp_path / "artifacts" / "test_model.pkl"

    metrics = run_training_pipeline(
        model_name="random_forest",
        model_path=artifact_path,
    )

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

    assert artifact_path.exists()
    assert artifact_path.is_file()

    assert settings.evaluation_reports_docs.exists()
