import time

import numpy as np
import pandas as pd

from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.pickle_model import JoblibWildfireModel
from prodml.utils.config import settings


def benchmark_model(model, X: pd.DataFrame) -> tuple[float, float]:
    """Measure mean and p95 prediction latency."""

    latencies = []

    for index in range(len(X)):
        sample = X.iloc[[index]]

        start = time.perf_counter()

        model.predict_one(sample)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    return mean_latency, p95_latency


def main() -> None:
    """Benchmark Pickle and ONNX models on the same validation data."""

    # Load the fixed 500-row validation dataset.
    validation_data = pd.read_csv(settings.validation_data_path)

    X = validation_data[list(settings.feature_names)]

    if len(X) != 500:
        raise ValueError(f"Expected 500 validation rows, found {len(X)}.")

    # Load both models.
    pickle_model = JoblibWildfireModel(settings.model_path)

    onnx_model = ONNXWildfireModel(
        settings.artifacts_dir / f"{settings.model_name}.onnx"
    )

    # Benchmark Pickle.
    pickle_mean, pickle_p95 = benchmark_model(
        pickle_model,
        X,
    )

    # Benchmark ONNX.
    onnx_mean, onnx_p95 = benchmark_model(
        onnx_model,
        X,
    )

    print("=" * 60)
    print("MODEL INFERENCE BENCHMARK")
    print("=" * 60)

    print(f"Validation rows: {len(X)}")

    print("\nPickle")
    print(f"  Mean latency: {pickle_mean:.3f} ms")
    print(f"  P95 latency:  {pickle_p95:.3f} ms")

    print("\nONNX")
    print(f"  Mean latency: {onnx_mean:.3f} ms")
    print(f"  P95 latency:  {onnx_p95:.3f} ms")

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
