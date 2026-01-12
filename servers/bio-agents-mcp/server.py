#!/usr/bin/env python3
"""
Protein Data Bank MCP Server
A simplified wrapper for the bio-agents-mcp PDB server following AI4S-agent-tools format
"""
import argparse
import sys
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Add the protein_data_bank_mcp source to Python path
sys.path.insert(0, str(Path(__file__).parent / "protein_data_bank_mcp" / "src"))

# Set default environment variables if not already set
os.environ.setdefault("PDB_MCP_HOST", "0.0.0.0")
os.environ.setdefault("PDB_MCP_PORT", "50001")

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Data Bank MCP Server")
    parser.add_argument("--port", type=int, default=50001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

args = parse_args()

# Update environment variables with command line arguments
os.environ["PDB_MCP_HOST"] = args.host
os.environ["PDB_MCP_PORT"] = str(args.port)

# Create FastMCP instance
mcp = FastMCP("protein-data-bank", host=args.host, port=args.port)

# Import all tools from the original server
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

# List of all tools
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

# Register all tools with our MCP instance
for tool in tools:
    mcp.tool()(tool)

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)