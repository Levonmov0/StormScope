"""AnomalyEvent and AnomalyReport models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.models.weather import WeatherDataPoint


class WeatherMetric(str, Enum):
    temperature = "temperature"
    precipitation = "precipitation"
    wind_speed = "wind_speed"
    wind_direction = "wind_direction"
    pressure = "pressure"
    humidity = "humidity"


class AnomalySeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AnomalyType(str, Enum):
    z_score = "z_score"
    iqr = "iqr"
    seasonal_deviation = "seasonal_deviation"


class AnomalyEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    metric: WeatherMetric
    value: float
    expected_range: tuple[float, float]
    severity: AnomalySeverity
    anomaly_type: AnomalyType


class AnomalyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: list[AnomalyEvent]
    cleaned_data: list[WeatherDataPoint]
    exclusion_recommendations: list[str]
