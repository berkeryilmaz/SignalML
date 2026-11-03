FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies needed for scientific Python packages and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and test suite
COPY backend/app /app/app
COPY backend/tests /app/tests

# Ensure directories for storage and host data exist
RUN mkdir -p /app/storage /data

# Define persistent volumes
VOLUME ["/app/storage", "/data"]

# Expose main application port
EXPOSE 8080

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8080
ENV SIGNALML_STORAGE_DIR=/app/storage
ENV SIGNALML_DATA_DIR=/data

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start Uvicorn serving FastAPI backend and frontend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--reload-dir", "/app/app"]
