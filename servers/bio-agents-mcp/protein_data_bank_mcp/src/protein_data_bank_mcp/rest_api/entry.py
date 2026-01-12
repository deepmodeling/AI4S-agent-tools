from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def structure(entry_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/entry/{entry_id}"
    result = await fetch_data(api_suffix)
    return result


async def pubmed_annotations(entry_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/pubmed/{entry_id}"
    result = await fetch_data(api_suffix)
    return result
