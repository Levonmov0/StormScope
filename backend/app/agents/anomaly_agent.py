"""Anomaly detection agent — result_type=AnomalyReport."""

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from app.models.anomaly import AnomalyEvent, AnomalyReport, WeatherMetric
from app.models.weather import WeatherDataPoint
from app.tools.anomaly_tools import (
    calculate_seasonal_deviations,
    detect_iqr_outliers,
    detect_zscore_anomalies,
    remove_anomalous_points,
)

_SYSTEM_PROMPT = """You are a weather anomaly detection system.

Your task:
1. Call `run_all_detections()` — this runs all three statistical detectors (z-score, IQR,
   seasonal deviation) across all six weather metrics and returns the combined anomaly events.
2. Analyze the results: identify patterns, note which metrics were affected, assess severities,
   and determine what data should be excluded from downstream forecasting.
3. Return a structured result with:
   - `events`: the full list of anomaly events returned by the tool (pass them through as-is)
   - `exclusion_recommendations`: clear, human-readable sentences explaining what anomalous
     patterns were found and why those data points should be excluded from forecasting.

Write exclusion recommendations as complete, informative sentences. For example:
"Exclude 3 high-severity temperature spikes in January 2024 — z-scores exceed 4.0 standard
deviations, indicating sensor malfunction or data entry errors."
"""


class _DetectionResult(BaseModel):
    """Internal agent result — events and recommendations only. Cleaning happens outside."""

    events: list[AnomalyEvent]
    exclusion_recommendations: list[str]


anomaly_agent: Agent[list[WeatherDataPoint], _DetectionResult] = Agent(
    "google-gla:gemini-2.0-flash",
    deps_type=list[WeatherDataPoint],
    result_type=_DetectionResult,
    system_prompt=_SYSTEM_PROMPT,
)


@anomaly_agent.tool
def run_all_detections(ctx: RunContext[list[WeatherDataPoint]]) -> list[dict]:
    """Runs all three detectors across all six weather metrics. Returns combined anomaly events."""
    events: list[AnomalyEvent] = []
    for metric in WeatherMetric:
        events.extend(detect_zscore_anomalies(ctx.deps, metric))
        events.extend(detect_iqr_outliers(ctx.deps, metric))
        events.extend(calculate_seasonal_deviations(ctx.deps, metric))
    return [e.model_dump(mode="json") for e in events]


async def run_anomaly_agent(data: list[WeatherDataPoint]) -> AnomalyReport:
    """Run the anomaly detection agent and return a complete AnomalyReport."""
    result = await anomaly_agent.run(
        "Analyze the provided weather data for anomalies across all six metrics.",
        deps=data,
    )
    detection = result.output
    cleaned = remove_anomalous_points(data, detection.events)
    return AnomalyReport(
        events=detection.events,
        cleaned_data=cleaned,
        exclusion_recommendations=detection.exclusion_recommendations,
    )
