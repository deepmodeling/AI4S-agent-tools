from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def structural_assembly_description(
    entry_id: str, assembly_id: str
) -> List[types.TextContent]:
    api_suffix = f"/core/assembly/{entry_id}/{assembly_id}"
    result = await fetch_data(api_suffix)
    return result
