"""Tests for analysis tools (trend comparison and event impact assessment)."""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType, WeatherMetric
from app.models.forecast import ForecastPoint, WeatherForecast
from app.models.weather import WeatherDataPoint
from app.tools.analysis_tools import (
    ImpactConfig,
    TrendConfig,
    _classify_direction,
    _classify_risk,
    assess_event_impacts,
    compare_trends,
)

_BASE_DT = datetime(2023, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_across_years(year_temps: dict[int, float]) -> list[WeatherDataPoint]:
    """One data point per year at the given temperature."""
    return [
        WeatherDataPoint(
            timestamp=datetime(year, 6, 15, tzinfo=timezone.utc),
            temperature=temp,
            precipitation=0.0,
            wind_speed=0.0,
            wind_direction=0.0,
            pressure=1013.0,
            humidity=50.0,
        )
        for year, temp in sorted(year_temps.items())
    ]


def _make_forecast(predicted_values: list[float]) -> WeatherForecast:
    """One ForecastPoint per day, starting from _BASE_DT."""
    predictions = [
        ForecastPoint(
            timestamp=_BASE_DT + timedelta(days=i),
            predicted_value=v,
            confidence_lower=v - 1.0,
            confidence_upper=v + 1.0,
        )
        for i, v in enumerate(predicted_values)
    ]
    return WeatherForecast(
        metric=WeatherMetric.temperature,
        predictions=predictions,
        seasonal_factors={},
        accuracy_mae=0.0,
    )


def _make_event(
    metric: WeatherMetric = WeatherMetric.temperature,
    severity: AnomalySeverity = AnomalySeverity.low,
) -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=_BASE_DT,
        metric=metric,
        value=30.0,
        expected_range=(10.0, 20.0),
        severity=severity,
        anomaly_type=AnomalyType.z_score,
    )


# ---------------------------------------------------------------------------
# compare_trends — historical direction
# ---------------------------------------------------------------------------


def test_compare_trends_rising_yearly():
    data = _make_data_across_years({2020: 10.0, 2021: 15.0})
    result = compare_trends(data, _make_forecast([10.0, 10.0]))
    assert result["trend_rate"] == pytest.approx(5.0)
    assert result["historical_direction"] == "rising"


def test_compare_trends_falling_yearly():
    data = _make_data_across_years({2020: 15.0, 2021: 10.0})
    result = compare_trends(data, _make_forecast([10.0, 10.0]))
    assert result["trend_rate"] == pytest.approx(-5.0)
    assert result["historical_direction"] == "falling"


def test_compare_trends_flat_yearly():
    data = _make_data_across_years({2020: 15.0, 2021: 15.0})
    result = compare_trends(data, _make_forecast([10.0, 10.0]))
    assert result["trend_rate"] == pytest.approx(0.0)
    assert result["historical_direction"] == "stable"


def test_compare_trends_single_year_only():
    """All data in one year → no year-over-year change possible → stable."""
    data = _make_data_across_years({2021: 15.0})
    result = compare_trends(data, _make_forecast([10.0, 10.0]))
    assert result["trend_rate"] == pytest.approx(0.0)
    assert result["historical_direction"] == "stable"


def test_compare_trends_fewer_than_min_data_points():
    """Single data point → early return with all-stable defaults."""
    data = _make_data_across_years({2021: 15.0})
    config = TrendConfig(min_data_points=2)
    result = compare_trends(data, _make_forecast([10.0, 10.0]), config=config)
    assert result["trend_rate"] == pytest.approx(0.0)
    assert result["historical_direction"] == "stable"
    assert result["forecast_direction"] == "stable"


# ---------------------------------------------------------------------------
# compare_trends — forecast direction
# (values chosen so the annualized delta clearly exceeds the 0.5 °C/year threshold)
# ---------------------------------------------------------------------------


def test_compare_trends_rising_forecast():
    """10 → 20 over 30 days annualizes to ~121 °C/year → rising."""
    result = compare_trends(
        _make_data_across_years({2020: 10.0, 2021: 10.0}),
        _make_forecast([10.0] + [10.0] * 28 + [20.0]),  # 30 predictions, span = 29 days
    )
    assert result["forecast_direction"] == "rising"


def test_compare_trends_falling_forecast():
    """20 → 10 over 30 days → falling."""
    result = compare_trends(
        _make_data_across_years({2020: 10.0, 2021: 10.0}),
        _make_forecast([20.0] + [20.0] * 28 + [10.0]),
    )
    assert result["forecast_direction"] == "falling"


def test_compare_trends_single_prediction_stable():
    result = compare_trends(
        _make_data_across_years({2020: 10.0, 2021: 10.0}),
        _make_forecast([15.0]),
    )
    assert result["forecast_direction"] == "stable"


def test_compare_trends_empty_predictions_stable():
    result = compare_trends(
        _make_data_across_years({2020: 10.0, 2021: 10.0}),
        _make_forecast([]),
    )
    assert result["forecast_direction"] == "stable"


# ---------------------------------------------------------------------------
# _classify_direction — boundary behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, threshold, expected",
    [
        (0.5, 0.5, "stable"),    # exactly at boundary → stable (uses <=)
        (-0.5, 0.5, "stable"),
        (0.501, 0.5, "rising"),
        (-0.501, 0.5, "falling"),
    ],
)
def test_classify_direction_boundaries(value, threshold, expected):
    assert _classify_direction(value, threshold) == expected


# ---------------------------------------------------------------------------
# assess_event_impacts
# ---------------------------------------------------------------------------


def test_assess_empty_events():
    result = assess_event_impacts([])
    assert result["risk_level"] == "low"
    assert result["most_impacted_metric"] is None
    assert result["high_severity_count"] == 0


def test_assess_all_high_severity():
    events = [_make_event(severity=AnomalySeverity.high) for _ in range(5)]
    result = assess_event_impacts(events)
    assert result["risk_level"] == "high"


def test_assess_exactly_30_percent_high():
    """3 high out of 10 = 30% → boundary is inclusive → high risk."""
    events = [_make_event(severity=AnomalySeverity.high) for _ in range(3)] + [
        _make_event(severity=AnomalySeverity.low) for _ in range(7)
    ]
    result = assess_event_impacts(events)
    assert result["risk_level"] == "high"


def test_assess_exactly_10_percent_high():
    """1 high out of 10 = 10% → boundary is inclusive → medium risk."""
    events = [_make_event(severity=AnomalySeverity.high)] + [
        _make_event(severity=AnomalySeverity.low) for _ in range(9)
    ]
    result = assess_event_impacts(events)
    assert result["risk_level"] == "medium"


def test_assess_below_10_percent_high():
    """9 high out of 100 = 9% → low risk."""
    events = [_make_event(severity=AnomalySeverity.high) for _ in range(9)] + [
        _make_event(severity=AnomalySeverity.low) for _ in range(91)
    ]
    result = assess_event_impacts(events)
    assert result["risk_level"] == "low"


def test_assess_most_impacted_metric():
    events = (
        [_make_event(metric=WeatherMetric.temperature) for _ in range(3)]
        + [_make_event(metric=WeatherMetric.pressure) for _ in range(5)]
        + [_make_event(metric=WeatherMetric.humidity)]
    )
    result = assess_event_impacts(events)
    assert result["most_impacted_metric"] == "pressure"


def test_assess_severity_distribution():
    events = (
        [_make_event(severity=AnomalySeverity.high) for _ in range(2)]
        + [_make_event(severity=AnomalySeverity.medium) for _ in range(3)]
        + [_make_event(severity=AnomalySeverity.low) for _ in range(4)]
    )
    dist = assess_event_impacts(events)["severity_distribution"]
    assert dist == {"high": 2, "medium": 3, "low": 4}


def test_assess_all_keys_present_when_only_high():
    """Severity distribution must always include all three keys."""
    events = [_make_event(severity=AnomalySeverity.high) for _ in range(3)]
    dist = assess_event_impacts(events)["severity_distribution"]
    assert dist["medium"] == 0
    assert dist["low"] == 0


# ---------------------------------------------------------------------------
# _classify_risk — boundary behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.30, "high"),    # boundary → high
        (0.10, "medium"),  # boundary → medium
        (0.099, "low"),
    ],
)
def test_classify_risk_boundaries(ratio, expected):
    assert _classify_risk(ratio, ImpactConfig()) == expected
