#!/usr/bin/env python3
"""
GitHub Project Agent - 专门用于转换 GitHub 仓库到 AI4S MCP 格式
"""
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .github_tools import (
    explore_repository, read_file_content, analyze_mcp_structure,
    modify_file_content, create_standard_files, convert_repository_format,
    search_in_repository, list_directory_tree
)
from .github_prompt import GITHUB_AGENT_INSTRUCTION

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

github_agent = LlmAgent(
    name="GitHub_Project_Agent",
    model=LiteLlm(model=os.getenv('MODEL', 'gemini-2.0-flash')),
    description="专业的 GitHub 仓库转换助手，将 MCP 项目转换为 AI4S 标准格式",
    instruction=GITHUB_AGENT_INSTRUCTION,
    tools=[
        explore_repository,
        read_file_content,
        analyze_mcp_structure,
        modify_file_content,
        create_standard_files,
        convert_repository_format,
        search_in_repository,
        list_directory_tree
    ]
)

def create_github_agent():
    """创建并返回 GitHub Project Agent 实例"""
    return github_agent