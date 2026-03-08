"""Climate analysis agent — result_type=ClimateAnalysis."""

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from app.models.analysis import ClimateAnalysis
from app.models.anomaly import AnomalyReport
from app.models.forecast import WeatherForecast
from app.tools.analysis_tools import assess_event_impacts, compare_trends

_SYSTEM_PROMPT = """\
You are a climate analysis agent. Your job is to synthesize anomaly and forecast data into a
structured ClimateAnalysis. You must call both tools before returning.

Steps:
1. Call trend_comparison() — read historical_direction, forecast_direction, trend_rate.
2. Call event_impact_assessment() — read risk_level, most_impacted_metric, severity counts.
3. Return a ClimateAnalysis with these fields:

   - trend_direction: your judgment combining historical_direction and forecast_direction.
     If both agree, use that direction. If they conflict, prefer the historical direction
     unless the forecast strongly diverges.
   - trend_rate: the float from trend_comparison, passed through unchanged (°C/year).
   - risk_level: the string from event_impact_assessment, passed through unchanged.
   - most_impacted_metric: the string from event_impact_assessment, or null if no events.
   - health_assessment: 2–3 sentences naming the location (if known), synthesizing trend,
     risk level, and key findings. Example:
     "Stockholm saw 16 anomalies concentrated in January; temperatures are trending upward
     at 0.4 °C/year and the 30-day forecast confirms continued warming."
   - event_summary: 1–2 sentences on which metrics were most affected and at what severity.
     Example: "Temperature accounted for the majority of events, with 5 classified as high
     severity. Precipitation showed moderate anomalies concentrated in summer months."

Ground every claim in tool result numbers. Do not invent figures.
"""


class _AnalysisInput(BaseModel):
    """Bundles both upstream agent outputs as agent dependencies."""

    anomaly_report: AnomalyReport
    forecast: WeatherForecast


analysis_agent: Agent[_AnalysisInput, ClimateAnalysis] = Agent(
    "google-gla:gemini-2.0-flash",
    deps_type=_AnalysisInput,
    result_type=ClimateAnalysis,
    system_prompt=_SYSTEM_PROMPT,
)


@analysis_agent.tool
def trend_comparison(ctx: RunContext[_AnalysisInput]) -> dict:
    """Compare historical yearly temperature trend against the 30-day forecast direction."""
    return compare_trends(ctx.deps.anomaly_report.cleaned_data, ctx.deps.forecast)


@analysis_agent.tool
def event_impact_assessment(ctx: RunContext[_AnalysisInput]) -> dict:
    """Count anomaly events per metric, find the most impacted, and classify overall risk."""
    return assess_event_impacts(ctx.deps.anomaly_report.events)


async def run_analysis_agent(
    anomaly_report: AnomalyReport,
    forecast: WeatherForecast,
) -> ClimateAnalysis:
    """Run the climate analysis agent and return a validated ClimateAnalysis."""
    deps = _AnalysisInput(anomaly_report=anomaly_report, forecast=forecast)
    result = await analysis_agent.run(
        "Analyse the anomaly report and forecast to produce a climate analysis.",
        deps=deps,
    )
    return result.output
