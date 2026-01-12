import json
import logging
import aiohttp
from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.constants import PDB_API_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def fetch_data(api_suffix: str) -> List[types.TextContent]:
    logger.info(f"Fetching data from {api_suffix}")
    url = f"{PDB_API_URL}{api_suffix}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_json = await response.json()
    result = types.TextContent(
        type="text", text=json.dumps(response_json, indent=2))
    return result
