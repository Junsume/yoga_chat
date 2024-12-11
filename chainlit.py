import chainlit as cl
from main import RAGChatbot

# Initialize the RAGChatbot
chatbot = RAGChatbot()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello! tell me how is your mood today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    response = chatbot.chat(query=message.content)
    response = response["result"]
    await cl.Message(content=response).send()