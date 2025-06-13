import chainlit as cl
from dotenv import load_dotenv
from rag import answer

# load .env in case you have other vars
load_dotenv()

@cl.on_chat_start
async def start():
    # Welcome message
    await cl.Message("👋 Hi! Ask me anything about Bhakkar & Layyah hospitals.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    q = message.content.strip()
    if not q:
        await cl.Message("Please ask a non-empty question.").send()
        return

    try:
        resp = answer(q)
    except Exception as e:
        resp = f"⚠️ Error: {e}"

    await cl.Message(resp).send()
