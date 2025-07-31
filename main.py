from agents import Agent , OpenAIChatCompletionsModel , AsyncOpenAI , Runner , RunConfig
from dotenv import load_dotenv
import chainlit as cl
import os 

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
 
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")

external_client = AsyncOpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config=RunConfig(
    model=model,
    model_provider=external_client
)

agent=Agent(
    name="Gemini Agent",
    instructions="You are a helfull assistant that can answer questions and help with tasks"
)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", []) 
    await cl.Message(content="Hello, how can i help you?").send()

@cl.on_message
async def handle_message(message:cl.Message):

    history=cl.user_session.get("history")

    history.append({"role":"user","content":message.content})
    result=await Runner.run(
        agent,
        input=history,
        run_config=config
    )
    history.append({"role":"assistant" , "content":result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()