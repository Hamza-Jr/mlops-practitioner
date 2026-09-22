# Module 1 — ML Production Foundations

## Model Evaluation Baseline

| Metric | Value |
|---|---:|
| Model | `baseline` |
| Model type | Random Forest Classifier |
| Validation accuracy | `80.54%` |
| Validation F1-score | `0.4218` |

- The baseline model is a Random Forest Classifier trained to classify wildfire
observations into `no_fire` (`0`) and `fire` (`1`).

## Serialization Benchmark

| Model | Mean latency | P95 latency |
|---|---:|---:|
| Pickle / Scikit-Learn | `32.970 ms` | `47.926 ms` |
| ONNX Runtime | `0.231 ms` | `0.263 ms` |

- Benchmark performed on the same fixed 500-row validation dataset.

## Serialization Comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No | Yes |
| Protobuf | No | Yes | Yes | Yes |
| Pickle | No | Limited | No | No |
| ONNX | No | Yes | Yes | Yes |

- The service serves the model in **ONNX** format because it provides a portable, cross-language format that can be executed with ONNX Runtime.


> Note: **Security note:** Pickle executes arbitrary code on load. Never load a `.pkl`
> file that was not produced by a trusted source.

## Serialization Decision

The service serves the **ONNX model** because it provides a portable model
format that can be executed with ONNX Runtime without requiring the original
Scikit-Learn model object.

## Parity Validation

The Pickle and ONNX models were executed against the same fixed 500-row
validation dataset.

- Prediction parity: **PASSED**
- Probability parity: **PASSED**
- Tolerance: `atol=1e-4`
