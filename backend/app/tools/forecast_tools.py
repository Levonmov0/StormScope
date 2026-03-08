"""Time-series forecasting functions: moving average, EMA, seasonal factors, forecast, backtest."""

from datetime import timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.models.anomaly import WeatherMetric
from app.models.forecast import ForecastPoint
from app.models.weather import WeatherDataPoint
from app.tools.anomaly_tools import _extract_values


class MovingAverageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_days: int = Field(7, ge=1)
    ema_span: int = Field(7, ge=1)
    min_data_points: int = Field(2, ge=1)


class SeasonalFactorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_data_points: int = Field(30, ge=1)


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_days: int = Field(30, ge=1)
    min_data_points: int = Field(2, ge=1)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    holdout_days: int = Field(30, ge=1)
    min_data_points: int = Field(60, ge=1)


DEFAULT_MOVING_AVERAGE = MovingAverageConfig()
DEFAULT_SEASONAL_FACTOR = SeasonalFactorConfig()
DEFAULT_FORECAST = ForecastConfig()
DEFAULT_BACKTEST = BacktestConfig()


def compute_moving_averages(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: MovingAverageConfig = DEFAULT_MOVING_AVERAGE,
) -> dict:
    if len(data) < config.min_data_points:
        return {}

    values = _extract_values(data, metric)
    timestamps = [dp.timestamp for dp in data]
    series = pd.Series(values, index=pd.DatetimeIndex(timestamps))

    sma = series.rolling(config.window_days).mean()
    ema = series.ewm(span=config.ema_span).mean()

    return {
        "timestamps": [t.isoformat() for t in timestamps],
        "sma": [None if np.isnan(v) else float(v) for v in sma],
        "ema": [float(v) for v in ema],
    }


def calculate_seasonal_factors(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: SeasonalFactorConfig = DEFAULT_SEASONAL_FACTOR,
) -> dict:
    if len(data) < config.min_data_points:
        return {"seasonal_factors": {}}

    values = _extract_values(data, metric)
    annual_mean = float(np.mean(values))

    df = pd.DataFrame({
        "month": [dp.timestamp.month for dp in data],
        "value": values,
    })

    monthly_means = df.groupby("month")["value"].mean()
    seasonal_factors = {
        int(month): float(mean - annual_mean)
        for month, mean in monthly_means.items()
    }

    return {"seasonal_factors": seasonal_factors}


def forecast_next_days(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    mae: float,
    config: ForecastConfig = DEFAULT_FORECAST,
) -> dict:
    if len(data) < config.min_data_points:
        return {"predictions": []}

    ma_result = compute_moving_averages(data, metric)
    ema_values = ma_result.get("ema", [])
    last_ema = float(ema_values[-1]) if ema_values else 0.0

    seasonal_result = calculate_seasonal_factors(data, metric)
    seasonal_factors = seasonal_result.get("seasonal_factors", {})

    last_timestamp = max(dp.timestamp for dp in data)

    predictions = []
    for day in range(1, config.horizon_days + 1):
        future_dt = last_timestamp + timedelta(days=day)
        month = future_dt.month
        seasonal_offset = seasonal_factors.get(month, 0.0)
        predicted = last_ema + seasonal_offset

        predictions.append(ForecastPoint(
            timestamp=future_dt,
            predicted_value=predicted,
            confidence_lower=predicted - mae,
            confidence_upper=predicted + mae,
        ).model_dump(mode="json"))

    return {"predictions": predictions}


def accuracy_backtest(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: BacktestConfig = DEFAULT_BACKTEST,
) -> dict:
    if len(data) < config.min_data_points:
        return {"mae": 0.0}

    sorted_data = sorted(data, key=lambda dp: dp.timestamp)
    cutoff = sorted_data[-config.holdout_days].timestamp
    training = [dp for dp in sorted_data if dp.timestamp < cutoff]
    holdout = [dp for dp in sorted_data if dp.timestamp >= cutoff]

    if not training:
        return {"mae": 0.0}

    ma_result = compute_moving_averages(training, metric)
    ema_values = ma_result.get("ema", [])
    last_ema = float(ema_values[-1]) if ema_values else 0.0

    seasonal_result = calculate_seasonal_factors(training, metric)
    seasonal_factors = seasonal_result.get("seasonal_factors", {})

    errors = []
    for dp in holdout:
        month = dp.timestamp.month
        seasonal_offset = seasonal_factors.get(month, 0.0)
        predicted = last_ema + seasonal_offset
        actual = float(getattr(dp, metric.value))
        errors.append(abs(actual - predicted))

    mae = float(np.mean(errors)) if errors else 0.0
    return {"mae": mae}
