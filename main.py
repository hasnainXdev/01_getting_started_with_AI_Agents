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
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import random
import chainlit as cl

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


@function_tool(is_enabled=False)
def weather_tool(city = "new york"):
    return f"the weather in the {city} is sunny"


python_agent = Agent(
    name="Python Agent",
    instructions="you are a expert python agent that answer questions about python programming",
)

python_agent_as_tool = python_agent.as_tool(
    tool_name="Python_Agent_Tool",
    tool_description="you are a expert python agent that answer questions about python programming",
)

agent = Agent(
    name="Hasnain Assistent",
    instructions="you are my assistant, when you asked about python use `python_agent_tool` tool or if weather is asked use `weather_tool` tool",
    model=model,
    model_settings=ModelSettings(tool_choice="required"),
    tools=[weather_tool, python_agent_as_tool],
)


@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome to the Hasnain AI Assistent, What I can help you today!"
    ).send()


@cl.on_message
async def handle_massage(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(event.data.delta)
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history),
