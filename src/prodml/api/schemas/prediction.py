from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request schema for a single wildfire prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ndvi": 0.42,
                "lst": 14500.0,
                "burned_area": 5.0,
            }
        }
    )

    ndvi: float = Field(description="Normalized Difference Vegetation Index")
    lst: float = Field(description="Land Surface Temperature")
    burned_area: float = Field(description="Burned-area measurement")


class PredictionResponse(BaseModel):
    """Response schema for a single wildfire prediction."""

    class_name: str = Field(description="Predicted wildfire class: no_fire or fire")
    probability: float = Field(
        ge=0.0, le=1.0, description="Probability of the predicted class"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch wildfire prediction."""

    instances: list[PredictionRequest] = Field(
        min_length=1,
        description="List of wildfire observations to classify",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch wildfire prediction."""

    predictions: list[str] = Field(
        description="Predicted wildfire classes in input order"
    )
