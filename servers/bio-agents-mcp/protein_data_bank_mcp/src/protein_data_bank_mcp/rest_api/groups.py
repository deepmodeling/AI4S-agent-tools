from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def pdb_cluster_data_aggregation(group_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/entry_groups/{group_id}"
    result = await fetch_data(api_suffix)
    return result


async def aggregation_group_provenance(
    group_provenance_id: str,
) -> List[types.TextContent]:
    api_suffix = f"/core/group_provenance/{group_provenance_id}"
    result = await fetch_data(api_suffix)
    return result


async def pdb_cluster_data_aggregation_method(group_id: str) -> List[types.TextContent]:
    api_suffix = f"/core/polymer_entity_groups/{group_id}"
    result = await fetch_data(api_suffix)
    return result
