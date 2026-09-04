"""Information tools: weather and web search. Both are free and need no key."""

import requests

# WMO weather codes -> human words, so the LLM doesn't have to guess.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# Cities don't move, so looking one up twice is wasted seconds on a slow host.
_PLACES: dict = {}


def get_weather(city: str) -> dict:
    """Get the current weather and a two-day forecast for a city.

    Args:
        city: City name, e.g. "Pune" or "London".
    """
    # wttr.in takes a place name directly, so this is one request instead of a
    # geocode plus a forecast — and it answers from datacenters, which the
    # Open-Meteo free tier refuses to do reliably.
    try:
        data = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            headers={"User-Agent": "curl/8"},
            timeout=12,
        ).json()
        now = data["current_condition"][0]
        days = data["weather"][:2]
        return {
            "city": city,
            "current": {
                "temperature_c": now["temp_C"],
                "feels_like_c": now["FeelsLikeC"],
                "conditions": now["weatherDesc"][0]["value"],
                "humidity_percent": now["humidity"],
                "wind_kmph": now["windspeedKmph"],
            },
            "forecast": [
                {
                    "date": day["date"],
                    "max_c": day["maxtempC"],
                    "min_c": day["mintempC"],
                    "conditions": day["hourly"][4]["weatherDesc"][0]["value"],
                }
                for day in days
            ],
        }
    except Exception as exc:
        print(f"[weather] wttr.in failed for {city}: {exc}", flush=True)

    # Fall back to Open-Meteo if wttr.in is having a bad day.
    key = city.lower().strip()
    place = _PLACES.get(key)
    if place is None:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        results = geo.get("results") or []
        if not results:
            return {"error": f"Could not find a city named {city}"}
        place = results[0]
        _PLACES[key] = place

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "forecast_days": 2,
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    # Say so plainly when the service returns nothing useful, otherwise the
    # assistant tries to narrate empty readings and then re-searches the web.
    if forecast.get("error") or not forecast.get("current"):
        reason = forecast.get("reason", "no data returned")
        print(f"[weather] open-meteo gave nothing for {city}: {reason}", flush=True)
        return {
            "error": f"The weather service is unavailable ({reason}).",
            "next_step": "Use web_search for the weather instead.",
        }

    current = forecast.get("current", {})
    daily = forecast.get("daily", {})
    current["conditions"] = WEATHER_CODES.get(current.get("weather_code"), "unknown")
    daily["conditions"] = [
        WEATHER_CODES.get(code, "unknown") for code in daily.get("weather_code", [])
    ]
    return {
        "city": place["name"],
        "country": place.get("country"),
        "current": current,
        "daily_today_and_tomorrow": daily,
        "units": {"temperature": "celsius", "wind": "km/h"},
    }


def web_search(query: str) -> list:
    """Search the web for current information and return the top results.

    Args:
        query: The search query.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        return [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
            for r in ddgs.text(query, max_results=5)
        ]
