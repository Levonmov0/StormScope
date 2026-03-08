"""Open-Meteo HTTP client — fetches historical weather data, returns List[WeatherDataPoint]."""

from datetime import datetime, timedelta, timezone

import httpx

from app.models.weather import WeatherDataPoint

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = (
    "temperature_2m,"
    "precipitation,"
    "windspeed_10m,"
    "winddirection_10m,"
    "pressure_msl,"
    "relativehumidity_2m"
)

PRESET_LOCATIONS: dict[str, tuple[float, float]] = {
    "Stockholm": (59.33, 18.07),
    "London":    (51.51, -0.13),
    "New York":  (40.71, -74.01),
    "Tokyo":     (35.68, 139.69),
    "Sydney":    (-33.87, 151.21),
    "Cape Town": (-33.92, 18.42),
}


def _resolve_location(location: str) -> tuple[float, float]:
    if location not in PRESET_LOCATIONS:
        raise ValueError(f"Unknown location '{location}'. Available: {list(PRESET_LOCATIONS)}")
    return PRESET_LOCATIONS[location]


def _compute_date_range(years: int) -> tuple[str, str]:
    end = datetime.now(tz=timezone.utc).date() - timedelta(days=1)
    start = end.replace(year=end.year - years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _parse_response(data: dict) -> list[WeatherDataPoint]:
    hourly = data["hourly"]
    rows = zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"],
        hourly["windspeed_10m"],
        hourly["winddirection_10m"],
        hourly["pressure_msl"],
        hourly["relativehumidity_2m"],
    )
    points: list[WeatherDataPoint] = []
    for time, temp, precip, wind_speed, wind_dir, pressure, humidity in rows:
        if any(v is None for v in (time, temp, precip, wind_speed, wind_dir, pressure, humidity)):
            continue
        points.append(WeatherDataPoint(
            timestamp=datetime.fromisoformat(time).replace(tzinfo=timezone.utc),
            temperature=temp,
            precipitation=precip,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            pressure=pressure,
            humidity=humidity,
        ))
    return points


class OpenMeteoClient:
    """Fetches historical hourly weather data from the Open-Meteo archive API."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    async def fetch(self, location: str, years: int) -> list[WeatherDataPoint]:
        lat, lon = _resolve_location(location)
        start_date, end_date = _compute_date_range(years)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": HOURLY_VARIABLES,
            "timezone": "UTC",
        }
        response = await self._http.get(ARCHIVE_URL, params=params)
        response.raise_for_status()
        return _parse_response(response.json())
