"""WeatherDataPoint — raw ingest model from Open-Meteo."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

MIN_TEMPERATURE = -90.0
MAX_TEMPERATURE = 60.0
MIN_PRESSURE = 870.0
MAX_PRESSURE = 1084.0


class WeatherDataPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    temperature: float = Field(ge=MIN_TEMPERATURE, le=MAX_TEMPERATURE)
    precipitation: float = Field(ge=0.0)
    wind_speed: float = Field(ge=0.0)
    wind_direction: float = Field(ge=0.0, le=360.0)
    pressure: float = Field(ge=MIN_PRESSURE, le=MAX_PRESSURE)
    humidity: float = Field(ge=0.0, le=100.0)
