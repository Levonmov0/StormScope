"""Climate analysis functions: trend comparison and event impact assessment."""

from collections import Counter
from typing import TypedDict

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.models.anomaly import AnomalyEvent
from app.models.forecast import WeatherForecast
from app.models.weather import WeatherDataPoint

DAYS_PER_YEAR = 365.0


class TrendConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Both historical and forecast rates are normalized to °C/year before comparison,
    # so a single threshold is dimensionally consistent.
    stable_threshold: float = Field(0.5, gt=0)  # °C/year
    min_data_points: int = Field(2, ge=1)


class ImpactConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    high_risk_threshold: float = Field(0.30, gt=0)
    medium_risk_threshold: float = Field(0.10, gt=0)


DEFAULT_TREND = TrendConfig()
DEFAULT_IMPACT = ImpactConfig()


class TrendResult(TypedDict):
    historical_direction: str   # "rising" | "falling" | "stable"
    forecast_direction: str     # "rising" | "falling" | "stable"
    trend_rate: float           # °C/year (historical)


class ImpactResult(TypedDict):
    most_impacted_metric: str | None
    high_severity_count: int
    severity_distribution: dict[str, int]   # keys: "high", "medium", "low"
    risk_level: str                          # "high" | "medium" | "low"


def _classify_direction(rate: float, stable_threshold: float) -> str:
    """Classify a °C/year rate. Boundary (abs == threshold) is classified as stable."""
    if abs(rate) <= stable_threshold:
        return "stable"
    return "rising" if rate > 0 else "falling"


def _classify_risk(high_ratio: float, config: ImpactConfig) -> str:
    """Map the fraction of high-severity events to a risk level. Boundary → higher category."""
    if high_ratio >= config.high_risk_threshold:
        return "high"
    if high_ratio >= config.medium_risk_threshold:
        return "medium"
    return "low"


def compare_trends(
    cleaned_data: list[WeatherDataPoint],
    forecast: WeatherForecast,
    config: TrendConfig = DEFAULT_TREND,
) -> TrendResult:
    """Compare historical yearly temperature trend against the forecast direction.

    Both rates are expressed in °C/year so the same stable_threshold applies to both.

    Returns:
        historical_direction: trend inferred from year-over-year temperature means.
        forecast_direction: trend inferred from first-to-last forecast prediction,
            annualized by the actual timestamp span of the forecast.
        trend_rate: historical °C/year rate (first year mean → last year mean).
    """
    if len(cleaned_data) < config.min_data_points:
        return TrendResult(
            historical_direction="stable",
            forecast_direction="stable",
            trend_rate=0.0,
        )

    df = pd.DataFrame({
        "year": [dp.timestamp.year for dp in cleaned_data],
        "temperature": [dp.temperature for dp in cleaned_data],
    })
    yearly_means = df.groupby("year")["temperature"].mean()

    if len(yearly_means) < 2:
        trend_rate = 0.0
        historical_direction = "stable"
    else:
        years = sorted(yearly_means.index)
        # Simple first-year to last-year delta divided by elapsed years.
        # Partial boundary years (e.g. a December-only 2020 vs January-only 2024) can
        # skew this; in production, weight by sample count per year or use linear regression.
        trend_rate = float(yearly_means[years[-1]] - yearly_means[years[0]]) / (years[-1] - years[0])
        historical_direction = _classify_direction(trend_rate, config.stable_threshold)

    predictions = forecast.predictions
    if len(predictions) >= 2:
        raw_delta = predictions[-1].predicted_value - predictions[0].predicted_value
        days_span = (predictions[-1].timestamp - predictions[0].timestamp).days
        # Normalize to °C/year so the threshold is dimensionally consistent with trend_rate.
        # Guard against zero-span (all predictions share a timestamp) → treat as stable.
        annualized_delta = raw_delta * (DAYS_PER_YEAR / days_span) if days_span > 0 else 0.0
        forecast_direction = _classify_direction(annualized_delta, config.stable_threshold)
    else:
        forecast_direction = "stable"

    return TrendResult(
        historical_direction=historical_direction,
        forecast_direction=forecast_direction,
        trend_rate=trend_rate,
    )


def assess_event_impacts(
    events: list[AnomalyEvent],
    config: ImpactConfig = DEFAULT_IMPACT,
) -> ImpactResult:
    """Count anomaly events per metric, find the most impacted, and classify overall risk.

    Returns:
        most_impacted_metric: metric name with the highest event count, or None if no events.
        high_severity_count: number of events with severity == "high".
        severity_distribution: event counts for all three severity levels (always all three keys).
        risk_level: overall risk based on the fraction of high-severity events.
    """
    if not events:
        return ImpactResult(
            most_impacted_metric=None,
            high_severity_count=0,
            severity_distribution={"high": 0, "medium": 0, "low": 0},
            risk_level="low",
        )

    metric_counts = Counter(event.metric.value for event in events)
    most_impacted = metric_counts.most_common(1)[0][0]

    # Initialize all three keys so the distribution is always complete regardless of
    # which severities appear in the data.
    severity_distribution: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for event in events:
        severity_distribution[event.severity.value] += 1

    high_ratio = severity_distribution["high"] / len(events)

    return ImpactResult(
        most_impacted_metric=most_impacted,
        high_severity_count=severity_distribution["high"],
        severity_distribution=severity_distribution,
        risk_level=_classify_risk(high_ratio, config),
    )
