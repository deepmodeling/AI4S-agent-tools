# Protein Data Bank MCP Server

Micro-service for accessing the Protein Data Bank (PDB) API through the MCP protocol. This service provides structured access to protein structures, chemical components, and related annotations from the PDB database.

## Features

- Access to PDB core API endpoints
- Structured data retrieval for proteins, assemblies, and chemical components
- Support for annotation data from DrugBank, UniProt, and PubMed
- Built on FastMCP framework with async/await support
- Local PDB file storage and parsing capabilities

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables in `.env`:
```env
PDB_MCP_HOST=localhost
PDB_MCP_PORT=8080
```

## Available Tools

### Core Data
- `structural_assembly_description`: Get assembly structures
- `chemical_component`: Get chemical component details
- `polymer_entity`: Get polymer entity information
- `structure`: Get structure details

### Annotations
- `drugbank_annotations`: DrugBank data for compounds
- `uniprot_annotations`: UniProt protein annotations
- `pubmed_annotations`: PubMed literature references

### Repository Info
- `current_entry_ids`: List current PDB entries
- `structure_status`: Check entry status
- `unreleased_structures`: Get unreleased structure info

## Docker Usage

Run with Docker Compose:
```bash
docker compose up pdb-server
```

The server will be available at `http://localhost:8080`.
