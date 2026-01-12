import os
import logging

from dotenv import load_dotenv, find_dotenv
from mcp.server import FastMCP
from protein_data_bank_mcp.rest_api.assembly import structural_assembly_description
from protein_data_bank_mcp.rest_api.chemical_component import (
    chemical_component,
    drugbank_annotations,
)
from protein_data_bank_mcp.rest_api.entity_instance import (
    polymer_entity_instance,
    branched_entity_instance,
    non_polymer_entity_instance,
)
from protein_data_bank_mcp.rest_api.entity import (
    branched_entity,
    polymer_entity,
    non_polymer_entity,
    uniprot_annotations,
)
from protein_data_bank_mcp.rest_api.entry import structure, pubmed_annotations
from protein_data_bank_mcp.rest_api.groups import (
    aggregation_group_provenance,
    pdb_cluster_data_aggregation,
    pdb_cluster_data_aggregation_method,
)
from protein_data_bank_mcp.rest_api.interface import pairwise_polymeric_interface_description
from protein_data_bank_mcp.pdb_store.storage import get_residue_chains

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

mcp_server = FastMCP(os.environ["PDB_MCP_HOST"],
                     port=os.environ["PDB_MCP_PORT"])

tools = [
    structural_assembly_description,
    chemical_component,
    drugbank_annotations,
    polymer_entity_instance,
    branched_entity_instance,
    non_polymer_entity_instance,
    uniprot_annotations,
    branched_entity,
    polymer_entity,
    non_polymer_entity,
    structure,
    pubmed_annotations,
    aggregation_group_provenance,
    pdb_cluster_data_aggregation,
    pdb_cluster_data_aggregation_method,
    pairwise_polymeric_interface_description,
    get_residue_chains,
]

for tool in tools:
    mcp_server.tool()(tool)
