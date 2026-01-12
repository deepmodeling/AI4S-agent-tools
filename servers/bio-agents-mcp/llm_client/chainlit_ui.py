import chainlit as cl
import logging

from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize
from client import OllamaClient, MCPClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv(find_dotenv())
initialize(config_path="conf")
cfg = compose(config_name="config")

llm = OllamaClient(model=cfg.ollama.model, base_url=cfg.ollama.base_url)
client = MCPClient(llm)


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Protein Structure",
            icon="🧬",
            markdown_description="Query protein structures and assemblies from PDB.",
            starters=[
                cl.Starter(
                    label="Structure Info",
                    message="Can you get me the structure for 1HR7?",
                ),
                cl.Starter(
                    label="Assembly Info",
                    message="Can you get me the structural assembly description for 1HR7 assembly 1?",
                ),
                cl.Starter(
                    label="Residue Chains",
                    message="What are the residue chains in 1FAT?",
                ),
            ],
        ),
        cl.ChatProfile(
            name="Chemical Data",
            icon="⚗️",
            markdown_description="Query chemical compounds and drug data.",
            starters=[
                cl.Starter(
                    label="Chemical Component",
                    message="Can you get me information about the chemical component NAG?",
                ),
                cl.Starter(
                    label="DrugBank Annotations",
                    message="What are the DrugBank annotations for ATP?",
                ),
                cl.Starter(
                    label="Find Molecule",
                    message="Find information about the molecule Aspirin",
                ),
            ],
        ),
    ]


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that does API calls on various life science databases.",
            }
        ],
    )


@cl.on_message
async def main(message: cl.Message):
    await client.connect_to_servers(cfg.mcp.servers)

    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")

    max_history_length = 10
    stream = await client.process_query(message.content, history=message_history[-max_history_length:])

    async for part in stream:
        if token := part["message"]["content"] or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
