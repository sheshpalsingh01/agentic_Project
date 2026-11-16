import os
import logging

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import create_react_agent, AgentExecutor

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
logger = logging.getLogger(__name__)

st.set_page_config(page_title='ReAct AI Assistant', page_icon='🤖', layout='centered')

@st.cache_resource(show_spinner='Initializing AI agent...')
def build_agent():
    openai_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPEN_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    weather_key = os.getenv('WEATHERSTACK_API_KEY')
    missing = [name for name, value in [('OPENAI_API_KEY', openai_key), ('TAVILY_API_KEY', tavily_key), ('WEATHERSTACK_API_KEY', weather_key)] if not value]
    if missing:
        raise ValueError('Missing environment variables: ' + ', '.join(missing))
    os.environ['TAVILY_API_KEY'] = tavily_key

    search_tool = TavilySearchResults(max_results=2)

    @tool
    def get_current_weather(city: str) -> str:
        """Fetch current weather conditions for a specified city."""
        url = 'http://api.weatherstack.com/current'  # Use HTTPS if your plan supports it.
        try:
            response = requests.get(url, params={'access_key': weather_key, 'query': city}, timeout=15)
            response.raise_for_status()
            data = response.json()
            if 'current' not in data:
                return f"Could not fetch weather for {city}: {data.get('error', {}).get('info', 'Unknown API error')}"
            current = data['current']
            location = data['location']
            return (
                f"City: {location['name']}\n"
                f"Country: {location['country']}\n"
                f"Temperature: {current['temperature']}°C\n"
                f"Weather: {', '.join(current.get('weather_descriptions', []))}\n"
                f"Humidity: {current['humidity']}%\n"
                f"Wind speed: {current['wind_speed']} km/h\n"
                f"Feels like: {current['feelslike']}°C"
            )
        except requests.RequestException:
            logger.exception('Weather request failed')
            return 'Weather service is temporarily unavailable.'
        except (KeyError, TypeError, ValueError):
            logger.exception('Invalid weather response')
            return 'Weather service returned incomplete data.'

    tools = [search_tool, get_current_weather]
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0, api_key=openai_key)
    # Pulling a public prompt can deserialize untrusted objects; only use a trusted prompt.
    prompt = Client().pull_prompt('hwchase17/react', dangerously_pull_public_prompt=True)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, max_iterations=5)

st.title('🤖 ReAct AI Assistant')
st.caption('Ask questions, search the web, and check live weather.')

with st.sidebar:
    st.header('Tools')
    st.write('🔎 Tavily web search')
    st.write('🌤️ WeatherStack weather lookup')
    st.divider()
    st.caption('Model: GPT-4.1 mini · LangChain Classic ReAct')
    if st.button('🗑️ Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if not st.session_state.messages:
    st.info('Try: “What is the capital of India, and what is its current weather?”')

if question := st.chat_input('Ask me anything...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Thinking and using tools...'):
            try:
                executor = build_agent()
                # Classic ReAct prompt is single-turn; history is displayed in UI but not sent to the model.
                result = executor.invoke({'input': question})
                answer = str(result.get('output', 'No response returned.'))
            except Exception:
                logger.exception('Agent invocation failed')
                answer = 'Sorry, the agent could not complete your request. Check the terminal logs and API configuration.'
        st.markdown(answer)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
