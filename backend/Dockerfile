# Use slim Python base image (CPU inference — keeps image ~4GB vs 13GB)
FROM python:3.10-slim

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements (all pinned - fast resolver, no backtracking)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --retries 5 --timeout 300 --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 && \
    pip3 install --retries 5 --timeout 300 -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Set environment variables for huggingface and app paths
ENV DATA_DIR=/app/data
ENV CHROMA_DB_PATH=/app/chroma_db
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HUB_CACHE=/app/model_cache

# Run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "900", "--workers", "1", "app:app"]
