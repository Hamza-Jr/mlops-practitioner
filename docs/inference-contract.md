# Inference Contract

## 1.Purpose
This document defines the contract between the production application and
the trained wildfire classification model.
It specifies:

- the model artifact and model type
- the model input requirements
- the accepted input data types and shape
- the feature preprocessing requirements
- the output format and data types
- the wildfire class ordering
- the prediction logic
- the external inference/API contract

The goal is to ensure that production inference reproduces the input
processing and prediction behavior established during model development.
—
## 2.Feature Definitions

The model uses three environmental features:
- **NDVI:** Normalized Difference Vegetation Index, representing vegetation
  density/greenness.
  - Observed training-data range: `0.030735` to `0.781723`

- **LST:** Land Surface Temperature, representing the surface temperature
  measurement used in the dataset.
  - Observed training-data range: `13137.0` to `15611.570513`

- **BURNED_AREA:** Burned-area measurement representing the burned-area
  value used in the dataset.
  - Observed training-data range: `3.0` to `9.0`

These ranges describe the values observed in the training dataset. They are
not strict model input limits. Production inputs should use the same units,
definitions, and data representation as the training dataset.


## 3. Model Contract
The model contract defines the internal interface between the preprocessing
pipeline and the trained Random Forest model.

### 3.1 Model

- **Final production model:** `best_wildfire_model.pkl`
- **Model type:** Random Forest Classifier
- **Framework:** Scikit-learn
- **Number of output classes:** 2
 ```python
 	{
  	    0: "no_fire",
    	1: "fire"
  	}
 ```
- **Selection criterion:** Highest validation accuracy
- **Validation accuracy:** `80.54%`
---
### 3.2 Input

The model expects three numerical input features used during training.

#### Input Requirements
- **Number of features:** 3
- **Feature order:** `NDVI`, `LST`, `BURNED_AREA`
- **Feature values:** Numerical values
- **Typical data type:** `float64`
- **Missing values:** Not allowed
- **Infinite values:** Not allowed

#### Accepted Input Types
The Random Forest model accepts both:

- `pandas.DataFrame`
- `numpy.ndarray`
For a single observation, the input must contain one row and three
features.
**Input shape:** (1, 3)
**The general input shape for multiple observations is:**  (n_samples, 3)

### 3.3 Preprocessing
Before inference, the production application must:
- Validate that all required features are present.
- Validate that all values are numerical.
- Check for missing values.
- Check for infinite values.
- Preserve the feature order.
- Convert the input into the expected tabular numerical format.
- Pass the processed input directly to the Random Forest model.


### 3.4 Output
The Random Forest provides two prediction interfaces.
#### Class Prediction
The `predict()` method returns one predicted class for each input sample.
- Output type: `numpy.ndarray`
- Predicted classes: `0` or `1`
For a single observation:
- Output shape: `(1,)`
For a batch of `n` observations:
- Output shape: `(n,)`



#### Class Probability Prediction
The `predict_proba()` method returns the probability of each class.
- Output type: `numpy.ndarray`
For a single observation:
- Output shape: `(1, 2)`
For a batch of `n` observations:
- Output shape: `(n, 2)`

The probability columns follow the model's class ordering:
- Column 0 → `no_fire` probability
- Column 1 → `fire` probability
Probabilities are floating-point values in the range `[0, 1]`.
For each observation, the two probabilities sum to approximately `1.0`.
