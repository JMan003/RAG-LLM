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
# Take only torch itself from the PyTorch CPU index (--no-deps): newer pip rejects
# some packages mirrored there (e.g. typing_extensions) for inconsistent names.
# torch's own dependencies come from PyPI, listed explicitly below.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --retries 5 --timeout 300 --no-deps \
        --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 && \
    pip3 install --retries 5 --timeout 300 -r requirements.txt \
        filelock "typing-extensions>=4.8.0" "sympy==1.12" networkx jinja2 fsspec && \
    python3 -c "import torch, sentence_transformers, llama_index.core; print('torch', torch.__version__)"

# Copy application code
COPY app.py ingest.py ./
COPY evaluation ./evaluation

# Expose port
EXPOSE 5000

# Set environment variables for huggingface and app paths
ENV DATA_DIR=/app/data
ENV CHROMA_DB_PATH=/app/chroma_db
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HUB_CACHE=/app/model_cache

# Run the app with gunicorn. One worker (one model in memory) with several threads,
# so /health and /livez still answer while a long CPU generation is running;
# app.py serialises generations with a lock.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "900", "--workers", "1", "--threads", "4", "app:app"]
