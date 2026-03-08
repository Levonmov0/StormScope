"""Unit tests for app/clients/open_meteo.py — no network calls."""

from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pydantic

from app.clients.open_meteo import OpenMeteoClient, _parse_response, _resolve_location

VALID_API_RESPONSE = {
    "hourly": {
        "time":                ["2025-01-01T00:00", "2025-01-01T01:00"],
        "temperature_2m":      [5.0, 6.0],
        "precipitation":       [0.0, 0.1],
        "windspeed_10m":       [10.0, 12.0],
        "winddirection_10m":   [180.0, 200.0],
        "pressure_msl":        [1013.0, 1012.0],
        "relativehumidity_2m": [75.0, 78.0],
    }
}


def test_resolve_location_unknown_raises():
    with pytest.raises(ValueError, match="Unknown location"):
        _resolve_location("Atlantis")


def test_parse_response_valid_rows():
    points = _parse_response(VALID_API_RESPONSE)

    assert len(points) == 2
    first = points[0]
    assert first.temperature == 5.0
    assert first.precipitation == 0.0
    assert first.wind_speed == 10.0
    assert first.wind_direction == 180.0
    assert first.pressure == 1013.0
    assert first.humidity == 75.0
    assert first.timestamp.tzinfo == timezone.utc


def test_parse_response_skips_none_rows():
    data = {
        "hourly": {
            "time":                ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m":      [None, 6.0],
            "precipitation":       [0.0, 0.1],
            "windspeed_10m":       [10.0, 12.0],
            "winddirection_10m":   [180.0, 200.0],
            "pressure_msl":        [1013.0, 1012.0],
            "relativehumidity_2m": [75.0, 78.0],
        }
    }
    points = _parse_response(data)
    assert len(points) == 1


def test_parse_response_invalid_value_raises():
    data = {
        "hourly": {
            "time":                ["2025-01-01T00:00"],
            "temperature_2m":      [999.0],
            "precipitation":       [0.0],
            "windspeed_10m":       [10.0],
            "winddirection_10m":   [180.0],
            "pressure_msl":        [1013.0],
            "relativehumidity_2m": [75.0],
        }
    }
    with pytest.raises(pydantic.ValidationError):
        _parse_response(data)


async def test_fetch_returns_data_points():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=VALID_API_RESPONSE)

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    client = OpenMeteoClient(http_client=mock_http)
    points = await client.fetch("Stockholm", years=1)

    assert len(points) == 2
    mock_http.get.assert_awaited_once()
