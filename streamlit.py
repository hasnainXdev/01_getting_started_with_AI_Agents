import streamlit as st
from agents import (
    Agent,
    Runner,
    RunConfig,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

agent = Agent(
    name="GIAIC Agent",
    instructions="Hello! I'm your AI assistant.",
    model=model,
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 Hasnain AI Assistant")
st.write("Welcome to the Hasnain AI Assistant. How can I help you today?")


# ✅ Display full chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user message
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to chat
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Display streaming output container
    with st.chat_message("assistant"):
        placeholder = st.empty()

        async def get_response():
            full_response = ""
            result = Runner.run_streamed(
                agent,
                input=st.session_state.history,
                run_config=config,
            )

            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    full_response += event.data.delta
                    placeholder.markdown(full_response)

            st.session_state.history.append(
                {"role": "assistant", "content": result.final_output}
            )

        # Run async task in Streamlit
        asyncio.run(get_response())
