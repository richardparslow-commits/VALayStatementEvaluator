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
#
# Optional extras are deliberately NOT installed here, so the default image carries
# no cloud SDKs. Add the ones this deployment actually uses, or the corresponding
# feature reports a clear configuration error at runtime rather than failing at
# import:
#
#   requirements-backup.txt  audit log backup to s3/gcs/azure
#                            (filesystem destination needs nothing)
#   requirements-s3.txt      S3 blob store for large queued job payloads
#   requirements-otel.txt    OpenTelemetry tracing (VA_LSE_TRACING=1)
#
# The audit-backup CronJob runs from this same image, so an s3/gcs/azure
# destination needs the backup extras present *here*:
#
#   RUN pip install --no-cache-dir -r requirements-backup.txt
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code (narrow COPYs for better layer caching)
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY app/ ./app/
COPY run_app.py ./

# Create the logs directory (audit + diagnostic logs) and the blob directory
# (large job payloads shared with the workers — see app/blob_store.py).
#
# Both must exist and be owned by the runtime user in the *image*: a named volume
# or PVC mounted over a path that does not exist here is created root-owned, and
# the non-root process below would then be unable to write to it.
RUN mkdir -p /app/logs /app/blobs && chown -R nobody:nogroup /app/logs /app/blobs

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
