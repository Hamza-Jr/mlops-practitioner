# Model Architecture

## Purpose

- The goal of this architecture is to decouple wildfire prediction logic from the underlying model implementation.
- This allows the application to use different model types without changing the prediction logic.
- The current implementation uses a Pickle-based scikit-learn model, while the architecture also allows an ONNX Runtime implementation to be added later.


## Responsibilities

### ModelBase

Defines the common interface that every model implementation must follow.

It provides the contract for:

- Loading the trained model with `load()`.
- Running predictions for individual observations with `predict_one()`.
- Running predictions for batches with `predict_batch()`.
- Returning class probabilities with `predict_proba()`.

### Model Implementation

Responsible for:

- Loading the trained model artifact.
- Managing the loaded model internally.
- Running inference.
- Returning predictions.
- Returning class probabilities.

The current implementation uses a Pickle-based scikit-learn model.

### WildfirePredictor

Responsible for the higher-level prediction workflow.

It receives a model implementation through dependency injection and uses the common model interface to perform inference.

The predictor does not depend on a specific model implementation.


### Dependency Injection

The model is provided to the predictor:

- model = SomeModel(model_path)
- predictor = EmotionPredictor(model)

This allows the prediction logic to work with any implementation that follows ModelBase.

      WildfirePredictor
             │
             ▼
         ModelBase
             ▲
             │
 ┌───────────┴───────────┐
 │                       │
Pickle(RF,SVM,..)       ONNX



### Model Loading

The model is loaded when the concrete model object is initialized.

If loading fails, model initialization fails immediately instead of creating an unusable model object.

### Benefits

- Loose coupling — prediction logic is independent of the specific model implementation.

- Extensibility — additional model implementations can be introduced without changing the prediction logic.

- Testability — the predictor and model implementations can be tested independently.

- Maintainability — model-specific loading and inference logic remains isolated inside the model implementation.

- Clear responsibility — the model handles model operations while the predictor handles the higher-level prediction workflow.
