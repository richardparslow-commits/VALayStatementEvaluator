# Dockerfile — VA Lay Statement Evaluator
#
# Production image optimized for small footprint and fast startup.
# Build:  docker build -t va-lse:latest .
# Run:    docker run --env-file .env -p 8501:8501 -p 8001:8001 va-lse:latest

FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so logs appear immediately in docker logs / platform log drains.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies (zlib for PDF extraction via pypdf)
RUN apt-get update && \
    apt-get install -y --no-install-recommends zlib1g && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the hash-pinned lockfile
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code (narrow COPYs for better layer caching)
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY app/ ./app/
COPY run_app.py ./

# Create logs directory for audit + diagnostic logs
RUN mkdir -p /app/logs && chown -R nobody:nogroup /app/logs

# Run as non-root for security
USER nobody

EXPOSE 8501 8001

# Default command: start health sidecar + Streamlit
# run_app.py installs SIGTERM handlers and starts the health sidecar
# before handing off to Streamlit.
CMD ["streamlit", "run", "run_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
