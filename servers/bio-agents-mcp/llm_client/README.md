# LLM Client for Biological Data

A natural language interface for querying biological and chemical data through PDB and ChEMBL databases.

## Features

- Natural language queries to structured biological data
- Integration with PDB and ChEMBL through MCP protocol
- Two interface options:
  - Modern web UI (Chainlit)
  - Terminal-based interface

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- NVIDIA GPU (recommended) for Ollama

## Quick Start

1. Set environment variables in `.env`:
```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11435
PDB_MCP_HOST=localhost
PDB_MCP_PORT=8080
CHEMBL_MCP_HOST=localhost
CHEMBL_MCP_PORT=8081
```

2. Install dependencies:
```bash
uv sync
```

3. Start the services:
```bash
docker compose up -d
```

## Usage

### Web Interface

Start the Chainlit UI:
```bash
make run-chainlit
```
Then open http://localhost:8000 in your browser

### Terminal Interface

Run the command-line interface:
```bash
make run-client
```

## Configuration

Edit `conf/config.yaml` to configure:
- LLM model settings
- MCP server connections
- Tool configurations
