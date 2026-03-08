"""Statistical anomaly detection functions: z-score, IQR, seasonal deviation."""

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, ConfigDict, Field

from app.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType, WeatherMetric
from app.models.weather import WeatherDataPoint

# Minimum data points required for IQR (need Q1/Q3 to be meaningful)
_IQR_MIN_DATA_POINTS: int = 4


class ZScoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(3.0, gt=0)
    high_threshold: float = Field(4.0, gt=0)
    medium_threshold: float = Field(3.5, gt=0)
    min_data_points: int = Field(2, ge=1)


class IQRConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    multiplier: float = Field(1.5, gt=0)
    high_distance_ratio: float = Field(3.0, gt=0)
    medium_distance_ratio: float = Field(2.0, gt=0)


class SeasonalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    std_threshold: float = Field(2.0, gt=0)
    high_threshold: float = Field(3.0, gt=0)
    medium_threshold: float = Field(2.5, gt=0)
    min_data_points: int = Field(30, ge=1)


DEFAULT_ZSCORE = ZScoreConfig()
DEFAULT_IQR = IQRConfig()
DEFAULT_SEASONAL = SeasonalConfig()


def _extract_values(data: list[WeatherDataPoint], metric: WeatherMetric) -> np.ndarray:
    return np.array([getattr(dp, metric.value) for dp in data], dtype=float)


def _classify_zscore_severity(z_abs: float, config: ZScoreConfig) -> AnomalySeverity:
    if z_abs >= config.high_threshold:
        return AnomalySeverity.high
    if z_abs >= config.medium_threshold:
        return AnomalySeverity.medium
    return AnomalySeverity.low


def _classify_iqr_severity(distance: float, iqr: float, config: IQRConfig) -> AnomalySeverity:
    ratio = distance / iqr
    if ratio >= config.high_distance_ratio:
        return AnomalySeverity.high
    if ratio >= config.medium_distance_ratio:
        return AnomalySeverity.medium
    return AnomalySeverity.low


def _classify_seasonal_severity(deviation: float, config: SeasonalConfig) -> AnomalySeverity:
    if deviation >= config.high_threshold:
        return AnomalySeverity.high
    if deviation >= config.medium_threshold:
        return AnomalySeverity.medium
    return AnomalySeverity.low


def detect_zscore_anomalies(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: ZScoreConfig = DEFAULT_ZSCORE,
) -> list[AnomalyEvent]:
    if len(data) < config.min_data_points:
        return []

    values = _extract_values(data, metric)
    z_scores = stats.zscore(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    expected_range = (mean - config.threshold * std, mean + config.threshold * std)

    events = []
    for dp, z in zip(data, z_scores):
        z_abs = abs(float(z))
        if z_abs >= config.threshold:
            events.append(AnomalyEvent(
                timestamp=dp.timestamp,
                metric=metric,
                value=getattr(dp, metric.value),
                expected_range=expected_range,
                severity=_classify_zscore_severity(z_abs, config),
                anomaly_type=AnomalyType.z_score,
            ))
    return events


def detect_iqr_outliers(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: IQRConfig = DEFAULT_IQR,
) -> list[AnomalyEvent]:
    if len(data) < _IQR_MIN_DATA_POINTS:
        return []

    values = _extract_values(data, metric)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if iqr == 0:
        return []

    lower_fence = q1 - config.multiplier * iqr
    upper_fence = q3 + config.multiplier * iqr
    expected_range = (float(lower_fence), float(upper_fence))

    events = []
    for dp in data:
        value = getattr(dp, metric.value)
        if value < lower_fence or value > upper_fence:
            distance = min(abs(value - lower_fence), abs(value - upper_fence))
            events.append(AnomalyEvent(
                timestamp=dp.timestamp,
                metric=metric,
                value=value,
                expected_range=expected_range,
                severity=_classify_iqr_severity(float(distance), float(iqr), config),
                anomaly_type=AnomalyType.iqr,
            ))
    return events


def calculate_seasonal_deviations(
    data: list[WeatherDataPoint],
    metric: WeatherMetric,
    config: SeasonalConfig = DEFAULT_SEASONAL,
) -> list[AnomalyEvent]:
    if len(data) < config.min_data_points:
        return []

    df = pd.DataFrame({
        "timestamp": [dp.timestamp for dp in data],
        "value": [getattr(dp, metric.value) for dp in data],
        "month": [dp.timestamp.month for dp in data],
    })

    monthly_stats = df.groupby("month")["value"].agg(mean="mean", std="std")
    monthly_stats = monthly_stats[monthly_stats["std"] > 0]

    events = []
    for dp in data:
        month = dp.timestamp.month
        if month not in monthly_stats.index:
            continue

        monthly_mean = monthly_stats.loc[month, "mean"]
        monthly_std = monthly_stats.loc[month, "std"]
        value = getattr(dp, metric.value)
        deviation = abs(value - monthly_mean) / monthly_std

        if deviation >= config.std_threshold:
            expected_range = (
                monthly_mean - config.std_threshold * monthly_std,
                monthly_mean + config.std_threshold * monthly_std,
            )
            events.append(AnomalyEvent(
                timestamp=dp.timestamp,
                metric=metric,
                value=value,
                expected_range=expected_range,
                severity=_classify_seasonal_severity(float(deviation), config),
                anomaly_type=AnomalyType.seasonal_deviation,
            ))
    return events


def remove_anomalous_points(
    data: list[WeatherDataPoint],
    events: list[AnomalyEvent],
) -> list[WeatherDataPoint]:
    anomalous_timestamps = {e.timestamp for e in events}
    return [dp for dp in data if dp.timestamp not in anomalous_timestamps]
