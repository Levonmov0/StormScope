"""run_pipeline() — calls agents in sequence and assembles the final WeatherIntelligenceReport."""

from datetime import datetime, timezone

import httpx

from app.agents.analysis_agent import run_analysis_agent
from app.agents.anomaly_agent import run_anomaly_agent
from app.agents.forecast_agent import run_forecast_agent
from app.clients.open_meteo import OpenMeteoClient
from app.models.report import ReportMetadata, WeatherIntelligenceReport


async def run_pipeline(location: str, years: int) -> WeatherIntelligenceReport:
    async with httpx.AsyncClient() as http_client:
        client = OpenMeteoClient(http_client)
        raw_data = await client.fetch(location, years)

    anomaly_report = await run_anomaly_agent(raw_data)
    forecast = await run_forecast_agent(anomaly_report.cleaned_data)
    analysis = await run_analysis_agent(anomaly_report, forecast)

    metadata = ReportMetadata(
        location=location,
        generated_at=datetime.now(tz=timezone.utc),
        years_analyzed=years,
    )
    return WeatherIntelligenceReport(
        metadata=metadata,
        anomaly_report=anomaly_report,
        forecast=forecast,
        analysis=analysis,
    )
