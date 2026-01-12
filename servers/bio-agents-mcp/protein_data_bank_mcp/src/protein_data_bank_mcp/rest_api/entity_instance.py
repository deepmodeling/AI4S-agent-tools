from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def branched_entity_instance(
    entry_id: str, asym_id: str
) -> List[types.TextContent]:
    api_suffix = f"/core/branched_entity_instance/{entry_id}/{asym_id}"
    result = await fetch_data(api_suffix)
    return result


async def non_polymer_entity_instance(
    entry_id: str, asym_id: str
) -> List[types.TextContent]:
    api_suffix = f"/core/non_polymer_entity_instance/{entry_id}/{asym_id}"
    result = await fetch_data(api_suffix)
    return result


async def polymer_entity_instance(
    entry_id: str, asym_id: str
) -> List[types.TextContent]:
    api_suffix = f"/core/polymer_entity_instance/{entry_id}/{asym_id}"
    result = await fetch_data(api_suffix)
    return result
