# AXIS Docker Makefile

.PHONY: help build build-prod run stop clean logs ps

# Default target
help:
	@echo "AXIS Docker Commands:"
	@echo ""
	@echo "  make build          - Build development image"
	@echo "  make build-prod     - Build production image"
	@echo "  make run            - Run container (foreground)"
	@echo "  make start          - Run container in background"
	@echo "  make stop           - Stop running container"
	@echo "  make restart        - Restart container"
	@echo "  make clean          - Remove containers and images"
	@echo "  make logs           - View container logs"
	@echo "  make ps             - Show running containers"
	@echo "  make shell          - Shell into running container"
	@echo "  make health         - Check container health"
	@echo ""
	@echo "Docker Compose Commands:"
	@echo ""
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make restart-all    - Restart all services"

# Build development image
build:
	docker build -t axis:latest .

# Build production image
build-prod:
	docker build -f Dockerfile.production -t axis:production .

# Run container in foreground
run:
	docker run --rm -p 8000:8000 --name axis-server axis:latest

# Start container in background
start:
	docker run -d --name axis-server -p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/models:/app/models \
		axis:latest

# Stop running container
stop:
	docker stop axis-server || true
	docker rm axis-server || true

# Restart container
restart: stop start

# Remove containers and images
clean:
	docker stop axis-server axis-postgres axis-qdrant 2>/dev/null || true
	docker rm axis-server axis-postgres axis-qdrant 2>/dev/null || true
	docker rmi axis:latest axis:production 2>/dev/null || true

# View logs
logs:
	docker logs -f axis-server

# Show running containers
ps:
	docker ps --filter "name=axis"

# Shell into running container
shell:
	docker exec -it axis-server /bin/bash

# Check container health
health:
	@docker inspect --format='{{.State.Health.Status}}' axis-server 2>/dev/null || echo "Container not running"

# Docker Compose commands
up:
	docker-compose up -d

down:
	docker-compose down

restart-all:
	docker-compose restart

# Build with docker-compose
compose-build:
	docker-compose build

# Full rebuild
rebuild: clean compose-build up