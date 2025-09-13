FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy scheduler code
COPY scheduler/ ./scheduler/
COPY common/ ./common/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8003

CMD ["python", "-m", "scheduler.main"]
