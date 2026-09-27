from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    report_path: Path,
) -> dict[str, float | list[list[int]]]:
    """Evaluate a trained classification model and write a report."""

    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))

    f1 = float(f1_score(y_test, predictions, zero_division=0))

    matrix = confusion_matrix(y_test, predictions)

    model_type = model.__class__.__name__

    _write_report(
        report_path=report_path,
        model_name=model_name,
        model_type=model_type,
        accuracy=accuracy,
        f1=f1,
        confusion_matrix_values=matrix,
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": matrix.tolist(),
    }


def _write_report(
    report_path: Path,
    model_name: str,
    model_type: str,
    accuracy: float,
    f1: float,
    confusion_matrix_values: object,
) -> None:
    """Write model evaluation results to a Markdown report."""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluated_at = datetime.now(UTC).isoformat()

    report = f"""# Model Evaluation

**Model:** {model_name}
**Model Type:** {model_type}
**Evaluated at:** {evaluated_at}

## Metrics

- Accuracy: {accuracy:.4f}
- F1-Score: {f1:.4f}

## Confusion Matrix

```text
{confusion_matrix_values}

```
"""

    if report_path.exists():
        with report_path.open("a", encoding="utf-8") as file:
            file.write("\n\n---\n\n")
            file.write(report)
    else:
        report_path.write_text(
            report,
            encoding="utf-8",
        )
