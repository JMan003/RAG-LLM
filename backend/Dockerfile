# Use NVIDIA CUDA base image for GPU inference
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download models at build time so the container starts instantly
# This avoids OOM crashes and slow startup in the pod
RUN python3 -c "\
import os; \
os.makedirs('/app/model_cache', exist_ok=True); \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', cache_dir='/app/model_cache'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/model_cache'); \
print('All models downloaded successfully!')\
"

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
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "app:app"]
