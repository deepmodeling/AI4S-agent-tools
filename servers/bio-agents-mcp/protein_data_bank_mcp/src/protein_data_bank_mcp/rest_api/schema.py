from typing import List
from mcp import types
from protein_data_bank_mcp.rest_api.utils import fetch_data


async def assembly_schema() -> List[types.TextContent]:
    api_suffix = "/schema/assembly"
    return await fetch_data(api_suffix)


async def branched_entity_instance_schema() -> List[types.TextContent]:
    api_suffix = "/schema/branched_entity_instance"
    return await fetch_data(api_suffix)


async def branched_entity_schema() -> List[types.TextContent]:
    api_suffix = "/schema/branched_entity"
    return await fetch_data(api_suffix)


async def chem_comp_schema() -> List[types.TextContent]:
    api_suffix = "/schema/chem_comp"
    return await fetch_data(api_suffix)


async def drugbank_schema() -> List[types.TextContent]:
    api_suffix = "/schema/drugbank"
    return await fetch_data(api_suffix)


async def entry_schema() -> List[types.TextContent]:
    api_suffix = "/schema/entry"
    return await fetch_data(api_suffix)


async def nonpolymer_entity_instance_schema() -> List[types.TextContent]:
    api_suffix = "/schema/nonpolymer_entity_instance"
    return await fetch_data(api_suffix)


async def nonpolymer_entity_schema() -> List[types.TextContent]:
    api_suffix = "/schema/nonpolymer_entity"
    return await fetch_data(api_suffix)


async def polymer_entity_instance_schema() -> List[types.TextContent]:
    api_suffix = "/schema/polymer_entity_instance"
    return await fetch_data(api_suffix)


async def polymer_entity_schema() -> List[types.TextContent]:
    api_suffix = "/schema/polymer_entity"
    return await fetch_data(api_suffix)


async def pubmed_schema() -> List[types.TextContent]:
    api_suffix = "/schema/pubmed"
    return await fetch_data(api_suffix)


async def uniprot_schema() -> List[types.TextContent]:
    api_suffix = "/schema/uniprot"
    return await fetch_data(api_suffix)
