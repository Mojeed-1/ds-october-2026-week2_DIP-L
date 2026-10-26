from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HousingFeatures(BaseModel):
    """Input features accepted by the California housing model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY",
            }
        }
    )

    longitude: float = Field(description="Block longitude")
    latitude: float = Field(description="Block latitude")
    housing_median_age: float = Field(
        ge=0,
        description="Median age of houses in the block",
    )
    total_rooms: float = Field(ge=0, description="Total rooms in the block")
    total_bedrooms: float = Field(ge=0, description="Total bedrooms in the block")
    population: float = Field(ge=0, description="Block population")
    households: float = Field(ge=0, description="Number of households in the block")
    median_income: float = Field(gt=0, description="Median income for the block")
    ocean_proximity: Literal[
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN",
    ]


class HousingPrediction(BaseModel):
    """Predicted median house value returned by the API."""

    predicted_median_house_value: float = Field(
        ge=0,
        description="Predicted median house value in US dollars",
    )