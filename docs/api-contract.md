# API Contract

## Purpose

This document defines the API endpoints, request and response formats, and error handling for the wildfire classification API.

The API exposes the trained wildfire classification model through FastAPI.

The model uses three features:

- `ndvi`
- `lst`
- `burned_area`

Class mapping:

- `0` → `no_fire`
- `1` → `fire`

---

## Health Check

### Endpoint

GET `/health`

### Purpose

Check whether the API is ready for inference.

The endpoint returns a successful response only when the model is loaded in memory.

### Response

- HTTP 200 when the model is loaded.
- Returns an unhealthy response when the model is not loaded.

---

## Model Metadata

### Endpoint

GET `/metadata`

### Purpose

Return metadata about the loaded production model.

### Response

- HTTP 200
- Returns:
  - model version
  - training date
  - feature names
  - framework
  - artifact hash

---

## Wildfire Prediction

### Endpoint

POST `/predict`

### Purpose

Predict the wildfire class for a single observation.

### Request

- Content-Type: `application/json`
- Required fields:
  - `ndvi`
  - `lst`
  - `burned_area`
- All fields must be numerical.
- Missing and non-finite values are not allowed.

### Response

- HTTP 200
- Returns the predicted wildfire class.
- class : predicted class name
- probability : confidence/probability associated with the predicted class

---

## Batch Wildfire Prediction

### Endpoint

POST `/predict/batch`

### Purpose

Predict wildfire classes for multiple observations.

### Request

- Content-Type: `application/json`
- Accepts a list of observations.
- Each observation must contain:
  - `ndvi`
  - `lst`
  - `burned_area`

### Response

- HTTP 200
- Returns a list of predicted class names.
- The prediction order must match the input observation order.

---

## Error Handling

The API handles:

- Missing required fields — HTTP 422
- Invalid data types — HTTP 422
- Missing values — HTTP 422
- Infinite or non-finite values — HTTP 422
- Invalid request structure — HTTP 422
- Model or inference errors — HTTP 500

Validation errors must return a readable error message without exposing a stack trace.

Unexpected internal errors must return HTTP 500 without exposing internal implementation details.

---

## Input Features

The API uses the same features as the trained model:

- `ndvi` — float
- `lst` — float
- `burned_area` — float

The feature order passed to the model must remain:

1. `NDVI`
2. `LST`
3. `BURNED_AREA`

The observed training-data ranges are not strict API input limits.

---

## API Acceptance Criteria

The API implementation is considered complete when:

- `/health` returns HTTP 200 only when the model is loaded.
- `/metadata` returns the required model metadata.
- `/predict` successfully performs a single prediction.
- `/predict/batch` successfully performs batch predictions.
- Invalid inputs return HTTP 422 with readable validation errors.
- Unexpected errors return HTTP 500 without leaking internal details.
- The model is loaded once during application startup using FastAPI's lifespan mechanism.
- The model is not loaded inside request handlers.
