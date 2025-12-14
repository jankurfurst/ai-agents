"""
Simple MCP Server with example tools
Run with: uv run python mcp_server.py
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("demo-mcp-server")


# ============================================
# TOOLS DEFINITION
# ============================================

@mcp.tool()
def get_current_time(timezone: str = "local") -> str:
    """
    Get the current date and time.

    Args:
        timezone: Timezone name (e.g., 'UTC', 'Europe/Prague'). Default is local time.

    Returns:
        Current date and time string
    """
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Perform basic math calculations. Supports +, -, *, /, ** (power), % (modulo).

    Args:
        expression: Math expression to evaluate (e.g., '2 + 2', '10 * 5', '2 ** 8')

    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation - only allow basic math
        allowed_chars = set("0123456789+-*/.() %")
        if not all(c in allowed_chars for c in expression.replace(" ", "")):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get current weather for a city (demo - returns mock data).

    Args:
        city: City name (e.g., 'Prague', 'London', 'New York')

    Returns:
        Weather information for the city
    """
    # Mock weather data
    weather_data = {
        "prague": {"temp": 5, "condition": "Cloudy", "humidity": 75},
        "london": {"temp": 8, "condition": "Rainy", "humidity": 85},
        "new york": {"temp": 2, "condition": "Sunny", "humidity": 45},
        "tokyo": {"temp": 12, "condition": "Clear", "humidity": 60},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        w = weather_data[city_lower]
        return f"Weather in {city}:\n- Temperature: {w['temp']}°C\n- Condition: {w['condition']}\n- Humidity: {w['humidity']}%"
    else:
        # Default mock data for unknown cities
        return f"Weather in {city}:\n- Temperature: 10°C\n- Condition: Partly cloudy\n- Humidity: 65%"


@mcp.tool()
def text_stats(text: str) -> str:
    """
    Get statistics about a text (word count, character count, etc.).

    Args:
        text: Text to analyze

    Returns:
        Statistics about the text
    """
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    lines = len(text.split("\n"))
    sentences = text.count(".") + text.count("!") + text.count("?")

    return f"""Text Statistics:
- Words: {words}
- Characters (with spaces): {chars}
- Characters (without spaces): {chars_no_spaces}
- Lines: {lines}
- Sentences: {sentences}"""


# ============================================
# SERVER STARTUP
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Demo MCP Server (FastMCP)")
    print("="*60)
    print("Endpoint: http://localhost:8002/mcp")
    print("="*60)
    print("Available tools:")
    print("  - get_current_time: Get current date/time")
    print("  - calculate: Math calculations")
    print("  - get_weather: Weather info (mock)")
    print("  - text_stats: Text statistics")
    print("="*60)

    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8002)
