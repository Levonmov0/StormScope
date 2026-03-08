"""Tests for anomaly detection tools (z-score, IQR, seasonal deviation)."""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.anomaly import AnomalySeverity, WeatherMetric
from app.models.weather import WeatherDataPoint
from app.tools.anomaly_tools import (
    DEFAULT_IQR,
    DEFAULT_SEASONAL,
    DEFAULT_ZSCORE,
    _classify_iqr_severity,
    _classify_seasonal_severity,
    _classify_zscore_severity,
    calculate_seasonal_deviations,
    detect_iqr_outliers,
    detect_zscore_anomalies,
    remove_anomalous_points,
)

_BASE_DT = datetime(2023, 6, 1, tzinfo=timezone.utc)


def _point(timestamp: datetime, **fields) -> WeatherDataPoint:
    base = {
        "temperature": 15.0,
        "precipitation": 1.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "pressure": 1013.0,
        "humidity": 50.0,
    }
    base.update(fields)
    return WeatherDataPoint(timestamp=timestamp, **base)


_SERIES_BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _points_with_wind(values: list[float]) -> list[WeatherDataPoint]:
    return [
        _point(_SERIES_BASE + timedelta(hours=i), wind_speed=v)
        for i, v in enumerate(values)
    ]


# --- Z-score tests ---

def test_zscore_detects_obvious_spike():
    normal_values = [10.0] * 99
    data = _points_with_wind(normal_values + [300.0])

    events = detect_zscore_anomalies(data, WeatherMetric.wind_speed)

    assert len(events) == 1
    assert events[0].value == 300.0
    assert events[0].severity == AnomalySeverity.high
    assert events[0].metric == WeatherMetric.wind_speed


def test_zscore_returns_empty_for_uniform_data():
    data = _points_with_wind([10.0] * 10)
    events = detect_zscore_anomalies(data, WeatherMetric.wind_speed)
    assert events == []


def test_zscore_insufficient_data_returns_empty():
    data = _points_with_wind([10.0])
    events = detect_zscore_anomalies(data, WeatherMetric.wind_speed)
    assert events == []


# --- IQR tests ---

def test_iqr_detects_outlier_beyond_fence():
    values = list(range(1, 11)) + [100]  # 1..10, 100
    data = _points_with_wind([float(v) for v in values])

    events = detect_iqr_outliers(data, WeatherMetric.wind_speed)

    assert len(events) == 1
    assert events[0].value == 100.0


def test_iqr_skips_zero_iqr():
    data = _points_with_wind([10.0] * 10)
    events = detect_iqr_outliers(data, WeatherMetric.wind_speed)
    assert events == []


def test_iqr_insufficient_data_returns_empty():
    data = _points_with_wind([1.0, 2.0, 3.0])  # < 4 points
    events = detect_iqr_outliers(data, WeatherMetric.wind_speed)
    assert events == []


# --- Seasonal deviation tests ---

def _seasonal_data() -> tuple[list[WeatherDataPoint], WeatherDataPoint]:
    """50 data points across January and July with one spike in January."""
    jan_points = [
        _point(datetime(2023, 1, d, tzinfo=timezone.utc), wind_speed=10.0)
        for d in range(1, 13)  # 12 points at 10.0
    ] + [
        _point(datetime(2023, 1, d, tzinfo=timezone.utc), wind_speed=11.0)
        for d in range(13, 25)  # 12 points at 11.0
    ]
    spike = _point(datetime(2023, 1, 25, tzinfo=timezone.utc), wind_speed=100.0)
    jan_points.append(spike)  # 25 Jan points total

    jul_points = [
        _point(datetime(2023, 7, d, tzinfo=timezone.utc), wind_speed=10.0)
        for d in range(1, 13)  # 12 points at 10.0
    ] + [
        _point(datetime(2023, 7, d, tzinfo=timezone.utc), wind_speed=11.0)
        for d in range(13, 26)  # 13 points at 11.0
    ]  # 25 Jul points total

    return jan_points + jul_points, spike


def test_seasonal_detects_month_deviation():
    data, spike = _seasonal_data()
    assert len(data) >= 30  # guard must pass

    events = calculate_seasonal_deviations(data, WeatherMetric.wind_speed)

    flagged_timestamps = {e.timestamp for e in events}
    assert spike.timestamp in flagged_timestamps


def test_seasonal_insufficient_data_returns_empty():
    data = _points_with_wind([10.0] * 10)  # < 30 points
    events = calculate_seasonal_deviations(data, WeatherMetric.wind_speed)
    assert events == []


# --- remove_anomalous_points tests ---

def test_remove_anomalous_points_excludes_flagged():
    spike_dt = datetime(2023, 6, 15, tzinfo=timezone.utc)
    normal = _points_with_wind([10.0] * 14)
    spike = _point(spike_dt, wind_speed=300.0)
    data = normal + [spike]

    events = detect_zscore_anomalies(data, WeatherMetric.wind_speed)
    cleaned = remove_anomalous_points(data, events)

    cleaned_timestamps = {dp.timestamp for dp in cleaned}
    assert spike_dt not in cleaned_timestamps
    assert len(cleaned) < len(data)


def test_remove_anomalous_points_empty_events_returns_all():
    data = _points_with_wind([10.0] * 5)
    cleaned = remove_anomalous_points(data, events=[])
    assert cleaned == data


# --- Severity boundary tests ---

def test_zscore_severity_at_exact_boundaries():
    assert _classify_zscore_severity(3.0, DEFAULT_ZSCORE) == AnomalySeverity.low
    assert _classify_zscore_severity(3.4999, DEFAULT_ZSCORE) == AnomalySeverity.low
    assert _classify_zscore_severity(3.5, DEFAULT_ZSCORE) == AnomalySeverity.medium
    assert _classify_zscore_severity(3.9999, DEFAULT_ZSCORE) == AnomalySeverity.medium
    assert _classify_zscore_severity(4.0, DEFAULT_ZSCORE) == AnomalySeverity.high


def test_iqr_severity_at_exact_boundaries():
    # distance/iqr ratios tested by passing distance=ratio, iqr=1.0
    assert _classify_iqr_severity(1.99, 1.0, DEFAULT_IQR) == AnomalySeverity.low
    assert _classify_iqr_severity(2.0, 1.0, DEFAULT_IQR) == AnomalySeverity.medium
    assert _classify_iqr_severity(2.99, 1.0, DEFAULT_IQR) == AnomalySeverity.medium
    assert _classify_iqr_severity(3.0, 1.0, DEFAULT_IQR) == AnomalySeverity.high


def test_seasonal_severity_at_exact_boundaries():
    assert _classify_seasonal_severity(2.49, DEFAULT_SEASONAL) == AnomalySeverity.low
    assert _classify_seasonal_severity(2.5, DEFAULT_SEASONAL) == AnomalySeverity.medium
    assert _classify_seasonal_severity(2.99, DEFAULT_SEASONAL) == AnomalySeverity.medium
    assert _classify_seasonal_severity(3.0, DEFAULT_SEASONAL) == AnomalySeverity.high
