"""WeatherIntelligenceReport and report Metadata models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.analysis import ClimateAnalysis
from app.models.anomaly import AnomalyReport
from app.models.forecast import WeatherForecast


class ReportMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    location: str
    generated_at: datetime
    years_analyzed: int


class WeatherIntelligenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: ReportMetadata
    anomaly_report: AnomalyReport
    forecast: WeatherForecast
    analysis: ClimateAnalysis
