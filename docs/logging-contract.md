# Logging Contract

## Purpose

The API uses structured JSON logs to record application events,
HTTP requests, and prediction activity.

The logs should help us answer:

- Is the API running?
- Was the model loaded?
- Which request was processed?
- How long did the request take?
- Did prediction succeed or fail?

## Log Format

Every log entry must be a single JSON object.

## Common Fields

Each log entry must contain:

- timestamp
- level
- logger
- message
- correlation_id

When applicable:

- event
- endpoint
- method
- status_code
- duration_ms
- model_version
- batch_size
- error_type

## Log Levels

- DEBUG — Detailed diagnostic information such as the feature vector.
- INFO — Normal application events and successful predictions with latency.
- WARNING — Expected abnormal conditions, such as input outside the observed training range.
- ERROR — Failures such as model loading failure or request validation rejection.

## Events

### Application

- application_started
- model_loaded
- application_shutdown

### HTTP

- http_request_completed

### Prediction

- prediction_requested
- prediction_successful
- prediction_failed

### Model

- model_load_failed

### Validation

- validation_rejected

### Monitoring

- input_outside_training_range

## Request Correlation ID

Every HTTP request must have a correlation ID.

The API should:

- accept an existing `X-Request-ID` header
- generate a UUID4 when it is not provided
- attach the correlation ID using Python `contextvars`
- include the correlation ID in related log entries
- return the correlation ID in the `X-Request-ID` response header

The same `correlation_id` must appear in:

- the request log
- the prediction log
- the response `X-Request-ID` header

## Prediction Logging

For `/predict`:

Log:

- prediction_requested
- prediction_successful
- prediction_failed

For `/predict/batch`:

Log:

- prediction_requested
- prediction_successful
- prediction_failed
- batch_size

Prediction success logs should include prediction latency.

Do not log the complete input payload.

## Input Range Warnings

Inputs outside the observed training-data range should generate a
WARNING log.

The observed training-data ranges are:

- `ndvi`: `0.030735` to `0.781723`
- `lst`: `13137.0` to `15611.570513`
- `burned_area`: `3.0` to `9.0`

These ranges are monitoring warnings and are not strict API input limits.

## Validation Errors

Invalid request data must generate an ERROR log with the event:

- validation_rejected

Validation errors must return HTTP 422 without exposing internal
implementation details or stack traces.

## Model Loading Errors

Model loading failures must generate an ERROR log with the event:

- model_load_failed

Internal implementation details and stack traces must not be exposed
to the API client.

## Logging Configuration

Structured logging must be configured in:

`src/prodml/logging_conf.py`

The implementation should use either:

- `python-json-logger`
- or a small custom `logging.Formatter`

Every emitted log line must be valid JSON.

## Security

Never log:

- secrets
- API keys
- tokens
- passwords
- complete request payloads
- unnecessary sensitive information

## Production Code

All `print()` statements must be removed from `src/`.

Application events must use the configured logging system.

## Example

{
  "timestamp": "2026-09-24T12:00:00Z",
  "level": "INFO",
  "logger": "prodml.api",
  "message": "Prediction served successfully",
  "event": "prediction_successful",
  "correlation_id": "8f4c7c3a-4f6d-4b0c-9a2e-123456789abc",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 4.21,
  "model_version": "0.1.0"
}

## Acceptance Check

Run the API with `curl` and inspect the logs using `jq`.

The logs must:

- be valid JSON
- contain `timestamp`
- contain `level`
- contain `logger`
- contain `message`
- contain `correlation_id`

The same `correlation_id` must appear in:

- the HTTP request log
- the prediction log
- the `X-Request-ID` response header
