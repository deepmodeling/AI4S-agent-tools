#!/usr/bin/env python3
"""
MCP工具生成器 - 通过对话收集需求并生成符合规范的MCP服务器
"""
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .tools import create_server, create_metadata, create_pyproject
from .prompt import INSTRUCTION

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

root_agent = LlmAgent(
    name="MCP_Agent",
    model=LiteLlm(model=os.getenv('MODEL', 'gemini-2.0-flash')),
    description="专业的MCP工具生成助手，帮助科学家快速创建MCP服务器",
    instruction=INSTRUCTION,
    tools=[create_server, create_metadata, create_pyproject]
)