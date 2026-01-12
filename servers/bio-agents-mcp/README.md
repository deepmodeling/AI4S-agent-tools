# Bio-Agents MCP

A collection of microservices and clients for natural language interaction with biological databases.

## Components

- **LLM Client**: Natural language interface with web UI and terminal modes
- **PDB MCP Server**: Protein Data Bank API service
- **ChEMBL MCP Server**: Chemical database API service

## Architecture

```
┌─────────────┐     ┌──────────────┐
│   LLM UI    │     │  Ollama LLM  │
│  (Chainlit) │     │              │
└─────┬───────┘     └───────┬──────┘
      │                     │
┌─────┴─────────────────────┴──────┐
│           LLM Client             │
└─────┬─────────────────────┬──────┘
      │                     │
┌─────┴───────┐     ┌──────┴───────┐
│  PDB MCP    │     │  ChEMBL MCP  │
│   Server    │     │    Server    │
└─────────────┘     └──────────────┘
```

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

## Development

- Use `make help` to see available commands
- Each service has its own README with detailed documentation
- Configuration files are in `conf/` directory

## Description

This project contains multiple modules that interact with various services and APIs using the FastMCP framework. Each module is designed to perform specific tasks and can be run independently or together using Docker Compose. The primary focus of this project is on bio agents, providing tools and services to interact with biological data sources such as the Protein Data Bank (PDB).

## Modules

### LLM Client

The `llm-client` module provides a client that interacts with a Language Model (LLM) server to process queries and utilize available tools. It is built using the FastMCP framework and supports asynchronous operations with `aiohttp`.

For more details, refer to the [LLM Client README](llm-client/README.md).

### Protein Data Bank

The `protein_data_bank_mcp` module provides a server that interacts with the Protein Data Bank (PDB) API to fetch structural assembly descriptions, chemical components, drugbank annotations, branched entities, non-polymer entities, polymer entities, uniprot annotations, structures, pubmed annotations, pdb cluster data aggregation, aggregation group provenance, pdb cluster data aggregation method, and pairwise polymeric interface descriptions. It is built using the FastMCP framework and supports asynchronous operations with `aiohttp`.

For more details, refer to the [Protein Data Bank README](protein-data-bank/README.md).

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
