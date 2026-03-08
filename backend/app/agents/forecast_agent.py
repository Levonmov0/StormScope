"""Weather forecast agent — result_type=WeatherForecast."""

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from app.models.anomaly import WeatherMetric
from app.models.forecast import ForecastPoint, WeatherForecast
from app.models.weather import WeatherDataPoint
from app.tools.forecast_tools import (
    accuracy_backtest,
    calculate_seasonal_factors,
    compute_moving_averages,
    forecast_next_days as _forecast_next_days,
)

_SYSTEM_PROMPT = """You are a weather forecasting system.

Your task:
1. Call `moving_averages()` to understand the smoothed EMA trend for temperature.
2. Call `seasonal_factors()` to get the absolute monthly offset from the annual mean.
3. Call `backtest_accuracy()` to compute the MAE on held-out data.
4. Call `forecast_next_days()` with the MAE from step 3 to generate 30-day predictions.
5. Return a structured result with:
   - `predictions`: the ForecastPoint list from forecast_next_days (pass through as-is)
   - `seasonal_factors`: the monthly dict from seasonal_factors (use integer keys)
   - `accuracy_mae`: the float MAE value from backtest
"""


class _ForecastResult(BaseModel):
    predictions: list[ForecastPoint]
    seasonal_factors: dict[int, float]
    accuracy_mae: float


forecast_agent: Agent[list[WeatherDataPoint], _ForecastResult] = Agent(
    "google-gla:gemini-2.0-flash",
    deps_type=list[WeatherDataPoint],
    result_type=_ForecastResult,
    system_prompt=_SYSTEM_PROMPT,
)


@forecast_agent.tool
def moving_averages(ctx: RunContext[list[WeatherDataPoint]]) -> dict:
    """Compute simple and exponential moving averages for temperature."""
    return compute_moving_averages(ctx.deps, WeatherMetric.temperature)


@forecast_agent.tool
def seasonal_factors(ctx: RunContext[list[WeatherDataPoint]]) -> dict:
    """Calculate the absolute monthly offset from the annual mean temperature."""
    return calculate_seasonal_factors(ctx.deps, WeatherMetric.temperature)


@forecast_agent.tool
def backtest_accuracy(ctx: RunContext[list[WeatherDataPoint]]) -> float:
    """Compute MAE by backtesting on held-out data."""
    result = accuracy_backtest(ctx.deps, WeatherMetric.temperature)
    return result["mae"]


@forecast_agent.tool
def forecast_next_days(ctx: RunContext[list[WeatherDataPoint]], mae: float) -> dict:
    """Generate 30-day temperature forecast using EMA trend + seasonal factors."""
    return _forecast_next_days(ctx.deps, WeatherMetric.temperature, mae)


async def run_forecast_agent(cleaned_data: list[WeatherDataPoint]) -> WeatherForecast:
    """Run the forecast agent and return a complete WeatherForecast."""
    result = await forecast_agent.run(
        "Analyze the cleaned weather data and generate a 30-day temperature forecast.",
        deps=cleaned_data,
    )
    r = result.output
    return WeatherForecast(
        metric=WeatherMetric.temperature,
        predictions=r.predictions,
        seasonal_factors=r.seasonal_factors,
        accuracy_mae=r.accuracy_mae,
    )
