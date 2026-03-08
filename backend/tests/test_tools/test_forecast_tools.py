"""Tests for forecast tools (moving average, EMA, seasonal factors, backtest, forecast)."""

from datetime import datetime, timedelta, timezone

from app.models.anomaly import WeatherMetric
from app.models.weather import WeatherDataPoint
from app.tools.forecast_tools import (
    DEFAULT_BACKTEST,
    DEFAULT_FORECAST,
    DEFAULT_MOVING_AVERAGE,
    DEFAULT_SEASONAL_FACTOR,
    BacktestConfig,
    SeasonalFactorConfig,
    accuracy_backtest,
    calculate_seasonal_factors,
    compute_moving_averages,
    forecast_next_days,
)

_BASE_DT = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _temp_series(values: list[float], base_dt: datetime = _BASE_DT) -> list[WeatherDataPoint]:
    return [
        WeatherDataPoint(
            timestamp=base_dt + timedelta(hours=i),
            temperature=v,
            precipitation=0.0,
            wind_speed=5.0,
            wind_direction=180.0,
            pressure=1013.0,
            humidity=50.0,
        )
        for i, v in enumerate(values)
    ]


def _multi_month_series(months_values: dict[int, list[float]]) -> list[WeatherDataPoint]:
    """Build a series with specific values per month (year 2023)."""
    points = []
    for month, values in months_values.items():
        for i, val in enumerate(values):
            points.append(WeatherDataPoint(
                timestamp=datetime(2023, month, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                temperature=val,
                precipitation=0.0,
                wind_speed=5.0,
                wind_direction=180.0,
                pressure=1013.0,
                humidity=50.0,
            ))
    return points


# --- compute_moving_averages ---

def test_moving_averages_returns_sma_and_ema():
    data = _temp_series([float(i) for i in range(20)])
    result = compute_moving_averages(data, WeatherMetric.temperature)

    assert "sma" in result
    assert "ema" in result
    assert "timestamps" in result
    assert len(result["sma"]) == len(data)
    assert len(result["ema"]) == len(data)


def test_moving_averages_insufficient_data_returns_empty():
    data = _temp_series([10.0])  # < min_data_points=2? No, 1 < 2
    result = compute_moving_averages(
        data, WeatherMetric.temperature, DEFAULT_MOVING_AVERAGE
    )
    assert result == {}


# --- calculate_seasonal_factors ---

def test_seasonal_factors_offsets_sum_near_zero():
    # 4 months, each with many uniform readings — offsets cancel against annual mean
    data = _multi_month_series({
        1: [-10.0] * 8,
        4: [5.0] * 8,
        7: [20.0] * 8,
        10: [5.0] * 8,
    })
    result = calculate_seasonal_factors(data, WeatherMetric.temperature)
    factors = result["seasonal_factors"]

    total = sum(factors.values())
    assert abs(total) < 1e-6


def test_seasonal_factors_hot_month_positive_cold_month_negative():
    # January cold, July hot
    data = _multi_month_series({
        1: [-15.0] * 8,
        7: [20.0] * 8,
    })
    # Need 30 points total
    extra = _multi_month_series({4: [5.0] * 14})
    result = calculate_seasonal_factors(
        data + extra, WeatherMetric.temperature
    )
    factors = result["seasonal_factors"]

    assert factors[7] > 0
    assert factors[1] < 0


def test_seasonal_factors_insufficient_data_returns_empty():
    data = _temp_series([15.0] * 10)  # < 30 points
    result = calculate_seasonal_factors(
        data, WeatherMetric.temperature, SeasonalFactorConfig(min_data_points=30)
    )
    assert result == {"seasonal_factors": {}}


# --- accuracy_backtest ---

def test_backtest_returns_mae():
    # 70 hourly points — enough for 60-point minimum
    data = _temp_series([float(i % 10) for i in range(70)])
    result = accuracy_backtest(data, WeatherMetric.temperature)
    assert result["mae"] >= 0.0


def test_backtest_insufficient_data_returns_zero_mae():
    data = _temp_series([15.0] * 30)  # < 60 points
    result = accuracy_backtest(
        data, WeatherMetric.temperature, BacktestConfig(min_data_points=60)
    )
    assert result["mae"] == 0.0


# --- forecast_next_days ---

def test_forecast_returns_correct_horizon():
    data = _temp_series([15.0] * 10)
    result = forecast_next_days(data, WeatherMetric.temperature, mae=1.0)
    assert len(result["predictions"]) == DEFAULT_FORECAST.horizon_days


def test_forecast_timestamps_are_in_future():
    data = _temp_series([15.0] * 10)
    last_ts = max(dp.timestamp for dp in data)

    result = forecast_next_days(data, WeatherMetric.temperature, mae=1.0)
    for point in result["predictions"]:
        # model_dump(mode="json") gives ISO string
        from datetime import datetime
        ts = datetime.fromisoformat(point["timestamp"])
        assert ts > last_ts


def test_forecast_confidence_band_brackets_prediction():
    data = _temp_series([15.0] * 10)
    result = forecast_next_days(data, WeatherMetric.temperature, mae=2.0)

    for point in result["predictions"]:
        assert point["confidence_lower"] <= point["predicted_value"]
        assert point["predicted_value"] <= point["confidence_upper"]


def test_forecast_confidence_band_width_equals_2_mae():
    mae = 3.5
    data = _temp_series([15.0] * 10)
    result = forecast_next_days(data, WeatherMetric.temperature, mae=mae)

    for point in result["predictions"]:
        width = point["confidence_upper"] - point["confidence_lower"]
        assert abs(width - 2 * mae) < 1e-9


def test_forecast_insufficient_data_returns_empty_predictions():
    data = _temp_series([15.0])  # 1 point < min_data_points=2
    result = forecast_next_days(data, WeatherMetric.temperature, mae=1.0)
    assert result == {"predictions": []}


def test_backtest_empty_training_returns_zero_mae():
    # holdout_days == len(data): all data is holdout, training is empty
    data = _temp_series([float(i) for i in range(10)])
    config = BacktestConfig(holdout_days=10, min_data_points=10)
    result = accuracy_backtest(data, WeatherMetric.temperature, config)
    assert result["mae"] == 0.0
