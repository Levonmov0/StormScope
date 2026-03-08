"""ClimateAnalysis model."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.models.anomaly import WeatherMetric


class TrendDirection(str, Enum):
    rising = "rising"
    falling = "falling"
    stable = "stable"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ClimateAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    health_assessment: str
    trend_direction: TrendDirection
    trend_rate: float
    risk_level: RiskLevel
    most_impacted_metric: WeatherMetric | None
    event_summary: str
