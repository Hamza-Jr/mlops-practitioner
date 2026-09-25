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

Each log entry should contain:

- timestamp
- level
- logger
- event

When applicable:

- request_id
- endpoint
- method
- status_code
- duration_ms
- model_version
- batch_size
- error_type

## Log Levels

- INFO — Normal application events
- WARNING — Expected abnormal conditions
- ERROR — Unexpected failures

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

## Request ID

Every HTTP request must have a request ID.

The API should:

- accept an existing `X-Request-ID` header
- generate a UUID when it is not provided
- include the request ID in the response header
- include the request ID in related log entries

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

Do not log the complete input payload.

## Security

Never log:

- secrets
- API keys
- tokens
- passwords
- complete request payloads
- unnecessary sensitive information

## Example

{
  "timestamp": "2026-09-24T12:00:00Z",
  "level": "INFO",
  "logger": "prodml.api",
  "event": "prediction_successful",
  "request_id": "8f4c7c3a-4f6d-4b0c-9a2e-123456789abc",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 4.21,
  "model_version": "0.1.0"
}
