# Dockerfile — VA Lay Statement Evaluator
#
# Two targets:
#
#   runtime (default)  the deployment image: smallest footprint, non-root, only
#                      the code the app needs. Unchanged by the sandbox below.
#   sandbox            an agent/dev workspace: same interpreter and same
#                      hash-pinned lock, plus the tooling and files a person or
#                      an agent needs to work in the box. See that stage.
#
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


# ---------------------------------------------------------------------------
# sandbox target — an agent/dev workspace, not a deployment artefact.
#
#   docker build --target sandbox -t va-lse-sandbox:latest .
#   vercel vcr build docker . va-lse-sandbox:latest --push
#   sandbox create --name va-lse-dev --image va-lse-sandbox:latest \
#     --vcpus 4 --timeout 2h --publish-port 8501 --connect
#
# A *custom* image rather than one of the managed sandbox images, and that is a
# dependency fact rather than a preference: those ship Python 3.14, and this
# lock has no 3.14 wheel for several of its pins (`jiter==0.17.0` is the first
# refusal), so a managed image cannot install the set CI proves. Building here
# means a sandbox and a deployment run the same interpreter and the same
# packages.
#
# Verifying what this stage produces is `tests/test_sandbox_image.py`'s job: no
# CI job builds an image, so without that test a missing COPY would surface only
# as a confusing failure inside a sandbox.
FROM runtime AS sandbox

# Root, because a workspace that cannot be written to is not a workspace: a
# clone, a .git directory and any generated log all need to land here.
USER root

# git   — the offline suite asks git for ground truth
#         (tests/test_security_gitignore.py runs `git check-ignore`), and an
#         agent needs clone/diff/commit in the box.
# curl  — every command in README → Health checks, and the API smoke tests.
# The rest is for navigating and debugging the code inside the box.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ripgrep less procps && \
    rm -rf /var/lib/apt/lists/*

# The dev extras, in the same order CI installs them (lock first, then these —
# see .github/workflows/test.yml). Additive: the lock's pins already satisfy
# requirements.txt's floors, so this brings in mypy/boto3/pyyaml/OpenTelemetry
# and moves nothing that CI proved.
RUN pip install --no-cache-dir -r requirements-dev.txt

# Everything the offline suite reads off the filesystem, so
# `python -m unittest discover -s tests` is green in a fresh sandbox: the tests
# and scripts themselves, the deploy assets (tests/test_monitoring_assets.py
# parses the compose file, the Prometheus rules and the Grafana dashboards), the
# workflow file read by tests/test_hermetic.py, and the pages whose links and
# anchors tests/test_docs_structure.py resolves. tests/test_sandbox_image.py
# fails when a root page or workflow file is added and not listed here.
COPY *.md ./
COPY .github/workflows/ ./.github/workflows/
# The Dockerfile itself: tests/test_sandbox_image.py reads it (it is how the
# contract in that file is checked), and it is the one file that explains to
# whoever opens the box how the box was built.
COPY Dockerfile ./
COPY .gitignore .env.example pyproject.toml docker-compose.yml ./
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/
COPY nginx/ ./nginx/
COPY examples/ ./examples/
COPY requirements*.txt ./

# A baseline commit, so `git status` in the sandbox shows the agent's own edits
# instead of the whole tree, and the suite's git-grounded test has a repository
# to ask. This is a snapshot of the image, not project history — when you have a
# real clone in the box, work from that.
RUN git init -q --initial-branch=main . && \
    git add -A && \
    git -c user.name=sandbox -c user.email=sandbox@localhost \
        commit -q -m "baseline: repository contents baked into the sandbox image"

# Sandbox ports are published as public URLs, and /health, /ready and /metrics
# carry no authentication (a kubelet cannot present a token), so the sidecar
# trusts only loopback here. To scrape them from outside, publish 8001 *and*
# override this to 0.0.0.0 — you are then publishing them to the internet.
# `sandbox exec curl -s localhost:8001/health` needs neither. README → Health
# checks explains the default and why the production image keeps it.
ENV VA_LSE_HEALTH_HOST=127.0.0.1

# No --server.port, unlike the runtime stage: 8501 is already Streamlit's default,
# and a *set* port is fatal when Streamlit resolves as a development-layout
# install — "server.port does not work when global.developmentMode is true"
# (.streamlit/config.toml documents the reproduction), which is exactly what a
# `pip install -e` or a vendored Streamlit in this box would be. Omitting it
# costs nothing and removes a way to wedge the app.
#
# Sandbox itself does not run ENTRYPOINT or CMD, so run this line by hand (or
# via `sandbox exec`) after boot; it is what `docker run` uses.
CMD ["streamlit", "run", "run_app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
