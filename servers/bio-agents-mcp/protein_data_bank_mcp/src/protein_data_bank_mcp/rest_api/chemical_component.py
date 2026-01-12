from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def chemical_component(comp_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/chemcomp/{comp_id}"
    result = await fetch_data(api_suffix)
    return result


async def drugbank_annotations(comp_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/drugbank/{comp_id}"
    result = await fetch_data(api_suffix)
    return result
