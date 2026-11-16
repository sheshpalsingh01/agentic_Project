import os
import certifi
import requests

from dotenv import load_dotenv
from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_classic.agents import (
    create_react_agent,
    AgentExecutor
)


# ==================================
# 1. Load Environment Variables
# ==================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is missing from .env")


# ==================================
# 2. Initialize Search Tool
# ==================================

search_tool = TavilySearchResults(max_results=2)


# ==================================
# 3. Initialize LLM
# ==================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# ==================================
# 4. Pull ReAct Prompt
# ==================================

client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)


# ==================================
# 5. Create Custom Weather Tool
# ==================================

@tool
def get_current_weather(city: str) -> str:
    """
    Fetch current weather information for a given city.
    """

    base_url = "http://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }

    try:
        res = requests.get(
            base_url,
            params=params,
            timeout=15
        )

        res.raise_for_status()
        data = res.json()

        if "current" not in data:
            return (
                f"Could not fetch weather data for {city}: "
                f"{data.get('error', {}).get('info', 'Unknown error')}"
            )

        current = data["current"]
        location = data["location"]

        return (
            f"City: {location['name']}\n"
            f"Country: {location['country']}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Weather: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind Speed: {current['wind_speed']} km/h\n"
            f"Feels Like: {current['feelslike']}°C"
        )

    except requests.exceptions.RequestException as e:
        return f"Weather API request failed: {str(e)}"

    except (KeyError, IndexError, TypeError) as e:
        return f"Error processing weather data: {str(e)}"


# ==================================
# 6. Register Tools
# ==================================

tools = [
    search_tool,
    get_current_weather
]


# ==================================
# 7. Create ReAct Agent
# ==================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ==================================
# 8. Create AgentExecutor
# ==================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


# ==================================
# 9. Execute Agent
# ==================================

response = agent_executor.invoke({
    "input": (
        "What is the capital of India? "
        "Then find the current weather of that capital."
    )
})

print(response["output"])