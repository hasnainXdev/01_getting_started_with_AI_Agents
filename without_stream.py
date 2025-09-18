from agents import (
    Agent,
    Runner,
    RunConfig,
    AsyncOpenAI,
    function_tool,
    OpenAIChatCompletionsModel,
    enable_verbose_stdout_logging,
    ModelSettings,
)
from agents.agent import StopAtTools
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import random
import chainlit as cl
import asyncio

load_dotenv()

enable_verbose_stdout_logging()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


@function_tool()
def weather_tool(city="new york"):
    return f"the weather in the {city} is sunny"


@function_tool
def greet_user(user="Hasnain"):
    return f"Hello! {user}"


agent = Agent(
    name="Hasnain Assistent",
    instructions="you are my assistant, use tools only when asked",
    model=model,
    model_settings=ModelSettings(
        tool_choice="required",
    ),
    tool_use_behavior=StopAtTools(stop_at_tool_names=["weather_tool", "greet_user"]),
    tools=[weather_tool, greet_user],
)


async def main():
    result = await Runner.run(
        agent,
        input="!",
        run_config=config,
    )

    print(result.final_output)


asyncio.run(main())
