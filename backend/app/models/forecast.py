"""ForecastPoint and WeatherForecast models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.anomaly import WeatherMetric


class ForecastPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float


class WeatherForecast(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: WeatherMetric
    predictions: list[ForecastPoint]
    seasonal_factors: dict[int, float]
    accuracy_mae: float
