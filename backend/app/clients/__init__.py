"""External API client integrations."""

from typing import Protocol, runtime_checkable

from app.models.weather import WeatherDataPoint


@runtime_checkable
class WeatherClient(Protocol):
    async def fetch(self, location: str, years: int) -> list[WeatherDataPoint]:
        """Fetch hourly weather data for a location over the given number of years."""
        ...
