from __future__ import annotations

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.utils.config import settings


def export_model(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Convert the trained sklearn model artifact to ONNX."""

    input_path = input_path or settings.model_path

    output_path = output_path or (
        settings.artifacts_dir / f"{settings.model_name}.onnx"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the existing trained model.
    model = joblib.load(input_path)

    # ONNX input shape: (n_samples, n_features)
    initial_type = [
        (
            "float_input",
            FloatTensorType([None, len(settings.feature_names)]),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
    )

    output_path.write_bytes(onnx_model.SerializeToString())

    return output_path


if __name__ == "__main__":
    export_model()
