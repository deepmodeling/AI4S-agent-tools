DOCKER_COMPOSE_FILE=docker-compose.yml

.PHONY: all
all: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup-env      - Set up the initial environment"
	@echo "  build          - Build all Docker images"
	@echo "  up             - Start all services using docker-compose"
	@echo "  down           - Stop all services using docker-compose"
	@echo "  restart        - Restart all services using docker-compose"
	@echo "  run-chainlit   - Run the Chainlit UI for the LLM client"
	@echo "  run-client     - Run the LLM client in terminal mode"

.PHONY: setup-env
setup-env:
	@echo "Setting up the initial environment..."
	uv sync --frozen --no-install-project --no-dev

.PHONY: build
build:
	@echo "Building all Docker images..."
	docker compose build

.PHONY: up
up:
	@echo "Starting all services..."
	docker compose up -d

.PHONY: down
down:
	@echo "Stopping all services..."
	docker compose down

.PHONY: restart
restart: down up

.PHONY: run-chainlit
run-chainlit:
	@echo "Starting Chainlit UI..."
	cd llm_client && chainlit run chainlit_ui.py

.PHONY: run-client
run-client:
	@echo "Starting LLM client in terminal mode..."
	cd llm_client && python client.py

.PHONY: targets
test:
	pytest
