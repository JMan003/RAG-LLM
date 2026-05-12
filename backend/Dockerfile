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

# Copy requirements
COPY requirements.txt .

# Install Python packages (CPU-only PyTorch to reduce image size from 13GB to ~4GB)
# --timeout 300: prevent read timeouts on slow network (default is 15s)
# --retries 5: retry failed downloads up to 5 times
RUN pip3 install --no-cache-dir \
    --timeout 300 \
    --retries 5 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Set environment variables for huggingface and app paths
ENV DATA_DIR=/app/data
ENV CHROMA_DB_PATH=/app/chroma_db
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "900", "--workers", "1", "app:app"]
