from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def current_entry_ids() -> List[types.TextContent]:
    api_suffix = "/holdings/current/entry_ids"
    result = await fetch_data(api_suffix)
    return result


async def structure_status(entry_id: str) -> List[types.TextContent]:
    api_suffix = f"/holdings/status/{entry_id}"
    result = await fetch_data(api_suffix)
    return result


async def structure_list_status(entry_ids: List[str]) -> List[types.TextContent]:
    api_suffix = f"/holdings/status?entry_ids={','.join(entry_ids)}"
    result = await fetch_data(api_suffix)
    return result


async def removed_structure_description(entry_id: str) -> List[types.TextContent]:
    api_suffix = f"/holdings/removed/{entry_id}"
    result = await fetch_data(api_suffix)
    return result


async def removed_entry_ids() -> List[types.TextContent]:
    api_suffix = "/holdings/removed/entry_ids"
    result = await fetch_data(api_suffix)
    return result


async def unreleased_structures(ids: List[str]) -> List[types.TextContent]:
    api_suffix = f"/holdings/unreleased?ids={','.join(ids)}"
    result = await fetch_data(api_suffix)
    return result


async def unreleased_structure_processing(entry_id: str) -> List[types.TextContent]:
    api_suffix = f"/holdings/unreleased/{entry_id}"
    result = await fetch_data(api_suffix)
    return result


async def unreleased_entry_ids() -> List[types.TextContent]:
    api_suffix = "/holdings/unreleased/entry_ids"
    result = await fetch_data(api_suffix)
    return result
