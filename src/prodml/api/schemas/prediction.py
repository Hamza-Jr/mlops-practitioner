from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class WildfireClass(str, Enum):
    """Valid wildfire prediction classes."""

    NO_FIRE = "no_fire"
    FIRE = "fire"


class PredictionRequest(BaseModel):
    """Request schema for a single wildfire prediction."""

    model_config = ConfigDict(
        extra="forbid",  # Reject unexpected keys
        json_schema_extra={
            "example": {
                "ndvi": 0.42,
                "lst": 14500.0,
                "burned_area": 5.0,
            }
        },
    )

    ndvi: float = Field(
        description="Normalized Difference Vegetation Index",
        allow_inf_nan=False,
    )
    lst: float = Field(
        description="Land Surface Temperature",
        allow_inf_nan=False,
    )
    burned_area: float = Field(
        description="Burned-area measurement",
        allow_inf_nan=False,
    )


class PredictionResponse(BaseModel):
    """Response schema for a single wildfire prediction."""

    class_name: WildfireClass = Field(
        description="Predicted wildfire class: no_fire or fire",
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class",
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
