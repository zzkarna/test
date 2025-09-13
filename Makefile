.PHONY: help build start stop test clean logs

help:
	@echo "Quant Dashboard MVP - Available commands:"
	@echo "  build    - Build all Docker images"
	@echo "  start    - Start all services"
	@echo "  stop     - Stop all services"
	@echo "  test     - Run test suite"
	@echo "  clean    - Clean up containers and volumes"
	@echo "  logs     - Show service logs"

build:
	docker-compose build

start:
	@echo "Starting Quant Dashboard..."
	@mkdir -p data/parquet data/duckdb logs
	docker-compose up -d
	@echo "Services starting..."
	@echo "Dashboard: http://localhost:8501"
	@echo "Analytics API: http://localhost:8002"

stop:
	docker-compose down

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

clean:
	docker-compose down -v
	docker system prune -f

logs:
	docker-compose logs -f

status:
	docker-compose ps
