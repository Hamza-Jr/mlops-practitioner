# MLOps Practitioner — Mini Project 1

A production-oriented machine learning service for wildfire classification using a Scikit-learn Random Forest model. The project takes the model from a research/baseline workflow into a modular Python package with a defined inference contract, model abstraction and dependency injection, ONNX serialization, FastAPI serving, structured JSON logging, request correlation IDs, automated tests, and Docker containerization. The API accepts three environmental features —`NDVI`, `LST`, and `BURNED_AREA` — and predicts either `fire` or `no_fire`

## Quick Start
1. Pull the published image
```bash
docker pull dimarco0/prodml-api:0.1.0
```

2. Run the API
```bash
docker run --rm -p 8000:8000 dimarco0/prodml-api:0.1.0
```

3. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ndvi": 0.42,
    "lst": 14500.0,
    "burned_area": 5.0
  }'
```

## The API is also available through Swagger UI at:
```bash
http://localhost:8000/docs
```
API endpoints
- **GET /health** — service health check

- **GET /metadata** — model metadata

- **POST /predict** — single prediction

- **POST /predict/batch** — batch predictions


## Project structure
```text
.
├── docker/
│   ├── .gitkeep
│   ├── Dockerfile
│   ├── Dockerfile.single-stage
│   └── docker-compose.yml
├── models/
│   └── artifacts/
│       ├── baseline.onnx
│       └── baseline.pkl
├── notebooks/
│   └── 00_baseline.ipynb
├── docs/
│   ├── api-contract.md
│   ├── inference-contract.md
│   ├── logging-contract.md
│   └── model-abstraction.md
├── reports/
│   ├── model_evaluation.md
│   └── module-1.md
├── scripts/
│   ├── benchmark_models.py
│   └── validate_onnx.py
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── py.typed
│       ├── api/
│       │   ├── __init__.py
│       │   ├── dependencies.py
│       │   ├── main.py
│       │   ├── middleware/
│       │   │   ├── __init__.py
│       │   │   └── correlation.py
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── health.py
│       │   │   ├── metadata.py
│       │   │   └── predict.py
│       │   └── schemas/
│       │       ├── __init__.py
│       │       ├── health.py
│       │       ├── metadata.py
│       │       └── prediction.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   ├── preprocessing.py
│       │   └── validation.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── context.py
│       │   └── logging_config.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── evaluate.py
│       │   ├── export.py
│       │   ├── factory.py
│       │   ├── onnx_model.py
│       │   ├── pickle_model.py
│       │   ├── predictor.py
│       │   └── train.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── data_pipeline.py
│       │   └── training_pipeline.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── decorators.py
│           └── hashing.py
├── tests/
│   ├── conftest.py
│   ├── test_serialization.py
│   ├── api/
│   │   ├── test_api.py
│   │   └── test_dependencies.py
│   ├── data/
│   │   ├── test_build_features.py
│   │   └── test_data_layer.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   └── test_training_pipeline.py
│   └── models/
│       ├── test_export.py
│       ├── test_predict.py
│       └── test_training.py
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```
## Docker Compose

```bash
docker compose -f docker/docker-compose.yml up
```

The Compose configuration mounts the model directory read-only and exposes the API on port `8000`.
