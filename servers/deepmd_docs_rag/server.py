
import sys
from pathlib import Path
import argparse

from fastmcp import FastMCP, Client
from typing import List
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()
FASTGPT_AUTH_TOKEN = os.getenv("FASTGPT_AUTH_TOKEN")

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DeePMD Docs RAG MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
backend = Client("https://zqibhdki.sealosbja.site/api/mcp/app/tRmg19AXvG2GL46rZXadjsIb/mcp")
mcp = FastMCP.as_proxy(backend, name="FastGPT Proxy Server", port=args.port, host=args.host)


@mcp.tool()
def upload_single_file_to_deepmd_knowledge_base(file_url: str):
    """get the file from the specified URL and upload it DeePMD-docs knowledge database.
    These files will be used to train the DeePMD-docs knowledge base.
    And future chat will use these documents to answer questions.
    """

    API_URL = "https://api.fastgpt.in/api/core/dataset/collection/create/localFile"
    DATASET_ID = "683a517f292645ebd7a2cb7c"      #  knowledge base: deepmd-from-chat

    # 1. download the file from the specified URL
    file_name = file_url.split("/")[-1].replace(" ", "_")
    file_content = requests.get(file_url).content
    
    # 2. upload the file to the specified dataset
    response = requests.post(
        url=API_URL,
        headers={"Authorization": f"Bearer {FASTGPT_AUTH_TOKEN}"},
        files={
            'file': (file_name, file_content),
            'data': (None, json.dumps({"datasetId": DATASET_ID, "parentId": None, "trainingType": "chunk", "chunkSize": 512}))
        }
    )
    
    print(f"Status: {response.status_code}, Response: {response.text}")
    return response

if __name__ == "__main__":
    mcp.run(transport="sse")
