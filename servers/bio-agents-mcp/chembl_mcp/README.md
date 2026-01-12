# ChEMBL MCP Server

Micro-service for accessing the ChEMBL database through the MCP protocol. This service provides natural language access to chemical compounds, drug data, and bioactivity information.

## Features

- Access to ChEMBL REST API endpoints
- Chemical compound and drug information retrieval
- Bioactivity data access
- Built on FastMCP framework with async/await support

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables in `.env`:
```env
CHEMBL_MCP_HOST=localhost
CHEMBL_MCP_PORT=8081
```

## Docker Usage

Run with Docker Compose:
```bash
docker compose up chembl-server
```

The server will be available at `http://localhost:8081`.

## Quick Start

1. Configure environment:
```bash
cp .env.example .env
```

2. Start services:
```bash
make build
make up
```

3. Launch web interface:
```bash
make run-chainlit
```

Visit http://localhost:8000 to start querying biological data.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (recommended) for optimal LLM performance
- Python 3.11+ (for local development)

## Example Queries

Try asking questions like:
- "What is the structure of protein 1HR7?"
- "Get me information about the chemical component NAG"
- "Show me the polymer entity for 7U7N entity 1"

## Development

- Use `make help` to see available commands
- Each service has its own README with detailed documentation
- Configuration files are in `conf/` directory

## Description

This project contains multiple modules that interact with various services and APIs using the FastMCP framework. Each module is designed to perform specific tasks and can be run independently or together using Docker Compose. The primary focus of this project is on bio agents, providing tools and services to interact with biological data sources such as the Protein Data Bank (PDB).

## Modules

### LLM Client

The `llm-client` module provides a client that interacts with a Language Model (LLM) server to process queries and utilize available tools. It is built using the FastMCP framework and supports asynchronous operations with `aiohttp`.

For more details, refer to the [LLM Client README](llm_client/README.md).

### Protein Data Bank

The `protein_data_bank_mcp` module provides a server that interacts with the Protein Data Bank (PDB) API to fetch structural assembly descriptions, chemical components, drugbank annotations, branched entities, non-polymer entities, polymer entities, uniprot annotations, structures, pubmed annotations, pdb cluster data aggregation, aggregation group provenance, pdb cluster data aggregation method, and pairwise polymeric interface descriptions. It is built using the FastMCP framework and supports asynchronous operations with `aiohttp`.

For more details, refer to the [Protein Data Bank README](protein-data-bank/README.md).

### ChEMBL Database

The `chembl_mcp` module provides a server that interacts with the ChEMBL API to access chemical compounds, drug data, and bioactivity information. It is built using the FastMCP framework and the ChEMBL web resource client.

For more details, refer to the [ChEMBL README](chembl_mcp/README.md).

## Docker

Dockerfiles are provided for each module to build Docker images.

- **Build the Docker image:**
  ```sh
  docker build -t <module-name> .
  ```

- **Run the Docker container:**
  ```sh
  docker run --env-file .env <module-name>
  ```

## Docker Compose

A `docker-compose.yml` file is provided to run all services together.

- **Start all services:**
  ```sh
  docker-compose up -d
  ```

- **Stop all services:**
  ```sh
  docker-compose down
  ```

## Makefile

A `Makefile` is provided to simplify common tasks.

- **Available targets:**
  - `setup-env`: Set up the initial environment.
  - `build`: Build all Docker images.
  - `up`: Start all services using docker-compose.
  - `down`: Stop all services using docker-compose.
  - `restart`: Restart all services using docker-compose.
  - `run-chainlit`: Run the Chainlit UI for the LLM client.
  - `run-client`: Run the LLM client in terminal mode.
