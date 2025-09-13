#!/bin/bash

# Start script for the Quant Dashboard
echo "Starting Quant Dashboard MVP..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Create data directories
mkdir -p data/parquet
mkdir -p data/duckdb
mkdir -p logs

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."
curl -f http://localhost:8002/health || echo "Analytics service not ready"
curl -f http://localhost:8001/health || echo "Dashboard service not ready"

echo "Quant Dashboard is starting up!"
echo "Dashboard: http://localhost:8501"
echo "Analytics API: http://localhost:8002"
echo "Logs: docker-compose logs -f"
