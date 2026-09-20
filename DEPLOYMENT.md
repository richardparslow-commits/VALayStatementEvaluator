# Deployment Guide — VA Lay Statement Evaluator

This document covers running the app in production at scale, including
multi-instance deployment behind a load balancer, running the pipeline on a
worker pool, and graceful failover. For single-user local setup, see
`README.md → Setup`.

## Table of contents

1. [Deployment patterns at a glance](#1-deployment-patterns-at-a-glance)
2. [Pattern A — Docker Compose + nginx (session affinity)](#2-pattern-a--docker-compose--nginx-session-affinity)
3. [Pattern B — Kubernetes with session affinity](#3-pattern-b--kubernetes-with-session-affinity)
4. [Pattern C — Kubernetes with a worker pool and Redis](#4-pattern-c--kubernetes-with-a-worker-pool-and-redis)
5. [Placement tradeoffs](#5-placement-tradeoffs)
6. [Dockerfile](#6-dockerfile)
7. [Health probe wiring](#7-health-probe-wiring)
8. [Graceful shutdown at scale](#8-graceful-shutdown-at-scale)
9. [Environment variables reference](#9-environment-variables-reference)
10. [Scaling guidance](#10-scaling-guidance)
11. [Rate limiting (reverse proxy)](#11-rate-limiting-reverse-proxy)
12. [TLS and reverse proxy](#12-tls-and-reverse-proxy)
13. [Distributed cache for VA reference data](#13-distributed-cache-for-va-reference-data)
14. [Pattern D — Streamlit Community Cloud](#14-pattern-d--streamlit-community-cloud)
15. [Distributed tracing (OpenTelemetry)](#15-distributed-tracing-opentelemetry)
16. [Audit log retention and backup](#16-audit-log-retention-and-backup)
17. [LLM endpoint failover (optional)](#17-llm-endpoint-failover-optional)

---

## 1. Deployment patterns at a glance

| Pattern | Instances | Where heavy runs execute | Failover | Complexity |
|---|---|---|---|---|
| **A. Docker Compose + nginx** | 3 (configurable) | In the pod serving the session | Affinity preserves session; new instance loses state | Low — ideal for small teams |
| **B. K8s + session affinity** | ≥ 2 via Deployment | In the pod serving the session | Same-node sessions survive pod restart; cross-node sessions lost | Medium |
| **C. K8s + worker pool + Redis** | ≥ 2 web + N workers | On a dedicated worker pool | Web pods are interchangeable; a killed worker's job is re-queued | High — best for 100-user scale |
| **D. Streamlit Community Cloud** | 1 (platform-managed) | In the single process | Platform restarts the app; in-memory state lost | None — no infrastructure (§14) |

**Recommendation for 100 concurrent users:** Pattern C (Kubernetes + worker pool).

Two things are true at once here, and only the second is fixed by adding Redis:

1. **Session affinity is still required.** A Streamlit session is a live WebSocket bound
to one server process; every rerun, widget event, and upload travels over that socket.
Replicating `st.session_state` to Redis does *not* let a different pod serve a user who is
connected elsewhere — if the load balancer routes `POST /_stcore/stream` to a pod that does
not hold the session, Streamlit answers "session not found → please reload". So Patterns
A/B/C all keep sticky sessions.

2. **But affinity must not decide where the work runs.** With the pipeline executing inside
the script run, a user with a 2,000-page bundle pins the pod that owns their socket for
~35 minutes and ~1.8 GB (`PERFORMANCE.md`), and a pod restart destroys all of it. Pattern C
moves the *run* — not the session — onto a worker pool. Web pods become interchangeable
and cheap; any pod can render a finished job's result; workers are scaled and sized
independently of the browser tier.

Patterns A/B are suitable for <20 users or development/staging. Redis is **required** in
Pattern C — not as a session store, but as the job queue, status channel, and result store.

**For a demo or single-user deployment with no infrastructure**, Pattern D (Streamlit
Community Cloud) is enough — but note it has no `VA_LSE_LOG_DIR` volume, so audit logs are
ephemeral. See [§14](#14-pattern-d--streamlit-community-cloud).

---

## 2. Pattern A — Docker Compose + nginx (session affinity)

This is the simplest multi-instance setup: three identical Streamlit containers
behind an nginx reverse proxy that pins users to a specific backend via a cookie.

### Quick start

```bash
# Build and start
docker compose up --build -d

# Scale to 5 instances (optional)
docker compose up -d --scale streamlit-web=5

# View logs
docker compose logs -f nginx
docker compose logs -f streamlit-web
```

#### Pattern C from the same compose file

The bundled compose file can also run the worker tier, so you can rehearse Pattern C
(including a real Redis) on one machine before touching a cluster:

```bash
# Web tier + Redis + 1 worker, with the shared job-document volume
VA_LSE_JOB_QUEUE=1 docker compose --profile pattern-c up --build -d

docker compose up -d --scale worker=4          # scale the worker tier
docker compose logs -f worker
docker compose exec redis redis-cli llen va_lse:jobs:evaluate
```

The `redis` and `worker` services sit behind the `pattern-c` profile, so a plain
`docker compose up` stays Pattern A. The worker reads the same `.env` as the web tier but has
no browser session — `OPENAI_API_KEY` must be in `.env` (or the `environment:` block), not
typed into the sidebar. Flip `VA_LSE_JOB_QUEUE=0` to go back to in-process runs; anything
already queued still finishes.

### `docker-compose.yml`

```yaml
services:
  nginx:
    image: nginx:1.27-alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - streamlit-web
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:80/nginx-health"]
      interval: 10s
      timeout: 3s
      retries: 3

  streamlit-web:
    build: .
    env_file: .env
    environment:
      - VA_LSE_HEALTH_PORT=8001
      - VA_LSE_SHUTDOWN_GRACE_SECONDS=30
      - VA_LSE_LLM_CALL_TIMEOUT_SECONDS=300
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2g
          cpus: "2.0"
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8001/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s
    read_only: true
    tmpfs:
      - /tmp:size=256M
```

### `nginx/nginx.conf`

The key directive is `ip_hash` for cookie-like session affinity. Each client IP
is pinned to a consistent backend. This works well when users come from different
IPs (typical for remote teams); for NAT-heavy networks, switch to the
`sticky cookie` method shown commented below.

```nginx
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    # --- Upstream: all Streamlit replicas ---
    upstream streamlit_backends {
        # ip_hash pins each client IP to a consistent backend.
        # For cookie-based sticky sessions (better behind NAT), uncomment:
        # sticky cookie srv_id expires=1h domain=.example.com path=/;
        ip_hash;

        server streamlit-web-1:8501;
        server streamlit-web-2:8501;
        server streamlit-web-3:8501;
    }

    # --- Liveness probe for nginx itself ---
    server {
        listen 80;
        server_name _;

        location /nginx-health {
            access_log off;
            return 200 '{"status":"ok","service":"nginx"}';
            add_header Content-Type application/json;
        }

        # --- All Streamlit traffic ---
        location / {
            proxy_pass http://streamlit_backends;

            # Required for Streamlit WebSocket to work through nginx
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts: streamlit can have long LLM calls
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;

            # Buffering off for streaming responses
            proxy_buffering off;
        }

        # --- Health sidecar pass-through ---
        # If you need direct access to individual instance health probes:
        # location /health {
        #     proxy_pass http://streamlit_backends/health;
        # }
    }
}
```

### Scaling notes

- **Replicas:** Change `replicas: 3` in `docker-compose.yml` and update the
  `nginx.conf` upstream block to match.
- **Resource limits:** The 2 GB memory limit is conservative for the default
  config. A single large evaluation run (2,000+ pages) can peak at ~1.5 GB
  during parallel digestion. For very large record sets, increase to 4 GB.
- **Health checks:** Docker Compose's healthcheck pings the `/health` sidecar.
  Unhealthy containers are removed from the upstream by the `depends_on`
  directive.

---

## 3. Pattern B — Kubernetes with session affinity

Kubernetes does not natively provide session affinity at the HTTP level.
Two options:

1. **Client-IP affinity** on the Service (`sessionAffinity: ClientIP`) — works
   when clients come from distinct IPs. Simple, but behind NAT/LB all traffic
   from one IP lands on one pod.
2. **Cookie-based affinity** via an Ingress controller (nginx-ingress, Traefik)
   — more reliable; the controller sets a `SRV_ID` cookie on first response.

### Manifests (`deploy/k8s/`)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: va-lse
  labels:
    app: va-lse
spec:
  replicas: 3
  selector:
    matchLabels:
      app: va-lse
  template:
    metadata:
      labels:
        app: va-lse
    spec:
      containers:
        - name: streamlit
          image: va-lse:latest
          ports:
            - containerPort: 8501
              name: http
            - containerPort: 8001
              name: health
          envFrom:
            - secretRef:
                name: va-lse-env
          resources:
            requests:
              memory: 1Gi
              cpu: "1.0"
            limits:
              memory: 3Gi
              cpu: "2.0"
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8001
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 10
      terminationGracePeriodSeconds: 45  # > VA_LSE_SHUTDOWN_GRACE_SECONDS (30) + buffer
```

```yaml
# k8s-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: va-lse
spec:
  selector:
    app: va-lse
  ports:
    - name: http
      port: 80
      targetPort: 8501
    - name: health
      port: 8001
      targetPort: 8001
  # Optional: Client-IP affinity (simple but NAT-unfriendly)
  # sessionAffinity: ClientIP
  # sessionAffinityConfig:
  #   clientIP:
  #     timeoutSeconds: 3600
```

### Ingress with sticky sessions (recommended for Pattern B)

```yaml
# k8s-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: va-lse
  annotations:
    # nginx-ingress sticky cookie configuration
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "VA_LSE_SESSION"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
    nginx.ingress.kubernetes.io/session-cookie-path: "/"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    # WebSocket support for Streamlit
    nginx.ingress.kubernetes.io/proxy-http-version: "1.1"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
spec:
  rules:
    - host: va-lse.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: va-lse
                port:
                  number: 80
```

### Deploy

```bash
# Create secret from .env
kubectl create secret generic va-lse-env --from-env-file=.env

# Apply manifests
kubectl apply -f deploy/k8s/

# Watch pods come up
kubectl get pods -l app=va-lse -w
```

---

## 4. Pattern C — Kubernetes with a worker pool and Redis

This is the recommended pattern for 100 concurrent users.

### How it works

The app ships a distributed job queue (`app/job_queue.py` + `app/worker.py`). With
`VA_LSE_JOB_QUEUE=1`, an Evaluate/Draft run is **submitted** instead of executed:

1. The web pod extracts the uploaded files (as it always has), serializes the statement
   or observations plus the extracted page text, and enqueues the job.
2. A worker pod claims it, runs `run_evaluation` / `run_draft` under the same memory and
   timeout guards as the in-process path, and writes the result back.
3. The web pod polls a small status key — a few hundred bytes per tick — and renders the
   report when the job finishes. Any web pod can render any job's result.

What this buys you over affinity alone:

| | Pattern A/B | Pattern C |
|---|---|---|
| Where a 2,000-page digest runs | the pod owning the user's WebSocket | a worker pod |
| Web-pod memory during a large run | ~1.8 GB peak same-process | a few MB (polling) |
| Pod killed mid-run | run lost, connection dropped | worker re-queues it; user reloads and re-attaches |
| User closes the tab mid-run | run is torn down (`BaseException` unwinds the script run) | run completes; results wait for the user |
| Scaling | web pods must be sized for the worst run | web and worker tiers scale independently |

**Sticky sessions stay on.** Affinity is not a workaround here — it is required by
Streamlit's WebSocket, and it is now cheap: what a user is stuck to is a thin UI pod.
Do **not** remove the Ingress affinity annotations unless you have replaced the
Streamlit server itself; a session that lands on the wrong pod gets "session not found".

**The worker needs the API key in its own environment.** A worker has no browser session,
so a key typed into the sidebar cannot reach it. Configure `OPENAI_API_KEY` (and the model
names) in the worker's env or mounted secret. The app checks this before queueing and
refuses the run with a clear message rather than failing on the worker.

### Redis

Redis holds the queue, per-job status, and results — not session state. Keys are namespaced
under `VA_LSE_JOB_QUEUE_PREFIX` (default `va_lse`):

| Key | Purpose |
|---|---|
| `va_lse:jobs:evaluate`, `va_lse:jobs:draft` | job ids waiting for a worker (list) |
| `va_lse:jobs:<kind>:leases` | heartbeat per running job (sorted set) — drives re-queue of abandoned jobs |
| `va_lse:job:<id>:meta` | status, progress, message, worker id (small JSON) |
| `va_lse:job:<id>:payload` | statement/observations + extracted record text — or a blob reference when large |
| `va_lse:job:<id>:result` | the finished report, usage totals |

All three carry `VA_LSE_JOB_QUEUE_TTL_SECONDS` (24 h default), so completed jobs expire on
their own.

Claims and worker updates use a shared Lua script (`EVAL`) on both Redis and
Upstash. Claiming records a unique attempt token and recovery lease before
removing queue membership. Recovery rechecks the heartbeat atomically; progress,
results, completion, failure, and draining requeues require the current token.
An interrupted request may have succeeded on the server: let its lease expire
and let the stale sweep recover it. Execution remains at-least-once, not
exactly-once. Recovery depends on retaining queue metadata and leases; database
loss, eviction, and payload expiration are not worker-interruption recovery.

**Upgrade:** drain/stop all old worker processes before starting this version.
Old workers do not enforce attempt ownership and must not overlap new workers.
Existing queued jobs and leased running jobs keep their key layout and remain
readable. Jobs already orphaned by an older version (absent from both the queue
and lease set) need operator reconciliation or resubmission; this change does
not scan all historical metadata. Redis credentials must permit `EVAL` and the
commands it executes. The supplied deployment uses standalone Redis; on Redis
Cluster, all queue keys must share a hash tag in `VA_LSE_JOB_QUEUE_PREFIX`
(for example `{va_lse}`). Changing that prefix creates a separate queue, so
drain the original queue before changing it.

Redis holds the *payload*, not the uploaded PDFs: extraction still happens on the web pod,
so what crosses the queue is the page-labelled record text. That text is what makes a large
job large, so above `VA_LSE_JOB_QUEUE_INLINE_MAX_BYTES` (256 KB default) the payload is
externalized — see **Job documents** below. Sizing follows from that: with externalization
on, Redis only needs room for small jobs, status, and results, and `--maxmemory` no longer
has to be sized for `concurrency × peak batch text` (a 5,000-page bundle serializes to
tens of megabytes, far past the 2 Gi the reference StatefulSet proposes).
`deploy/k8s/k8s-redis.yaml` provisions that StatefulSet; note that
`--maxmemory-policy allkeys-lru` will evict the largest keys first — which, once the big
payloads are in blob storage, are the finished results rather than a running job. Use
`noeviction` (or a larger `maxmemory`) if a lost job is worse than a failed write.

**Upstash instead of in-cluster Redis:** setting `VA_LSE_SHARED_CACHE_URL` +
`VA_LSE_SHARED_CACHE_TOKEN` (the credentials the reference cache already uses) makes the
same queue run over the Upstash REST API with no extra infrastructure and no `redis`
package. See README → Distributed cache for the setup steps. Upstash has no blocking pop, so
workers poll on `VA_LSE_JOB_QUEUE_POLL_SECONDS` (raise it to reduce request costs). Note that
Upstash bills per request and per byte, which is the other reason large payloads belong in
the blob store rather than in the queue.

### Job documents — inline or externalized

A job's documents go to a blob store when they exceed `VA_LSE_JOB_QUEUE_INLINE_MAX_BYTES`
(256 KB default); the queue then carries a small content-addressed reference instead. Small
jobs stay inline, so the zero-configuration path is unchanged and no extra round trip is
paid for a 40 KB statement.

| `VA_LSE_BLOB_STORE` | Backing store | Notes |
|---|---|---|
| `auto` (default) | filesystem when the queue is on, otherwise none | no configuration needed |
| `filesystem` | `VA_LSE_BLOB_DIR` (default `blobs`) | stdlib only; the directory **must be shared** with every worker |
| `s3` | `VA_LSE_BLOB_S3_BUCKET` (+ `_PREFIX`, `_ENDPOINT_URL`) | needs `pip install -r requirements-s3.txt`; works with AWS S3, Cloudflare R2, MinIO, DO Spaces |
| `none` | — | never externalize; every job must fit `VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES` |

The filesystem backend is the right default for a cluster because it needs no extra
service: `deploy/k8s/k8s-blobs.yaml` provisions a **ReadWriteMany** PVC mounted at
`/app/blobs` in both the web and worker Deployments, and the compose file mounts the same
`job-blobs` volume in both. A per-pod `emptyDir` (or a `ReadWriteOnce` PVC shared by
accident) fails in a specific, confusing way: the web pod writes the blob happily, then the
worker cannot resolve the reference and the job fails with `BlobNotFound`. The blob **must**
be visible to every pod that might claim the job, so verify with
`kubectl exec deploy/va-lse-worker -- ls /app/blobs` after a large run.

Use S3 when the web and worker tiers cannot share a filesystem — multi-cluster, or worker
nodes in another region. `deploy/k8s/k8s-deployment.yaml` / `k8s-worker.yaml` then need the
`AWS_*` credentials in the env secret, and no PVC.

Blobs are content-addressed (the key is derived from the bytes), so two users uploading the
same bundle in one deployment share one object, and re-submitting a failed run re-uses it.
They are deleted when the job reaches a terminal state, and the filesystem backend also
sweeps anything older than `VA_LSE_JOB_QUEUE_TTL_SECONDS` on write, so a crashed worker
cannot leak bytes forever. Storage needed is roughly `queue depth × average job text`;
a handful of concurrent 2,000-page bundles is a few hundred MB.

The sidebar's **🛠️ Job queue** panel and `GET /health → job_queue` report which blob
backend is active, so a misconfigured tier is visible before a job fails.

### Deployment

```bash
# Redis (or point the app at Upstash instead and skip this)
kubectl apply -f deploy/k8s/k8s-redis.yaml

# Credentials + queue flags
kubectl create secret generic va-lse-env --from-env-file=.env
kubectl patch secret va-lse-env -p \
  '{"data":{"VA_LSE_REDIS_URL":"cmVkaXM6Ly92YS1sc2UtcmVkaXM6NjM3OS8w","VA_LSE_JOB_QUEUE":"MQ=="}}'

# Web tier, then the worker tier
kubectl apply -f deploy/k8s/
kubectl rollout status deployment/va-lse-worker

# Confirm the queue is live and see the backlog
curl -s localhost:8001/health | jq .job_queue      # web pod
curl -s localhost:8002/health | jq .job_queue      # worker pod
```

Start workers **before** flipping `VA_LSE_JOB_QUEUE=1` on the web tier, otherwise users
queue jobs nothing is consuming. To roll back, unset it — in-flight jobs already on the
queue finish, and new runs go back in-process.

---

## 5. Placement tradeoffs

| Factor | Pattern A/B (run in the web pod) | Pattern C (worker pool) |
|---|---|---|
| **Setup complexity** | Low — just nginx cookie or k8s affinity annotation | High — Redis + worker Deployment + queue flags |
| **Where heavy work runs** | The pod serving the user's session | Dedicated worker pods |
| **Run survives pod restart** | No — the user re-uploads and re-runs | Yes — the job is re-queued and any web pod renders it |
| **Run survives a closed tab** | No | Yes |
| **Web pod memory** | Sized for the worst run (2,000 pages ≈ 1.8 GB) | Flat; workers carry the peak |
| **Scaling** | Vertical, or more identically-sized pods | Web and worker tiers scale independently |
| **Operational cost** | Zero additional infra | Redis + worker pods + monitoring |
| **When to use** | < 20 users, dev/staging, quick demos | 20–100+ concurrent users, production |

**For many deployments, Pattern B with session affinity is still sufficient** because pod
restarts are rare in a healthy cluster, and when one does happen the user refreshes and
re-runs (they already have the record bundle locally). The app stays fully functional
exactly as it always has.

Pattern C is worth the complexity when you need:
- Runs to survive pod restarts, rolling upgrades, and HPA scale-in
- Web pods that are interchangeable, so a browser session and a 2,000-page digest stop
  competing for the same pod's memory
- Users to be able to close the tab and come back to a finished report

---

## 6. Dockerfile

The production Dockerfile is optimized for the smallest possible image and
fastest startup. It installs from the hash-pinned lockfile and runs as a
non-root user.

What can reach either stage is decided by [`.dockerignore`](.dockerignore) as
well as by the `COPY` lines: it excludes credentials, the audit trail, the blob
store and exported reports from the build context entirely, so a later
`COPY . .` cannot pick them up either. See
[SECURITY.md §8](SECURITY.md#the-same-audit-for-dockerignore) for the guard that
keeps it that way.

```dockerfile
# Dockerfile
FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so logs appear immediately in docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for PDF extraction (pypdf uses zlib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends zlib1g && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the lockfile (hash-pinned)
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY app/ ./app/
COPY run_app.py ./
COPY scripts/ ./scripts/

# Create logs directory (audit + diagnostic logs)
RUN mkdir -p /app/logs && chown -R nobody:nogroup /app/logs

# Run as non-root
USER nobody

EXPOSE 8501 8001

# Default command: start health sidecar + Streamlit
# The launcher (run_app.py) installs SIGTERM handlers and starts the
# health sidecar before handing off to Streamlit.
CMD ["streamlit", "run", "run_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### Build

```bash
docker build -t va-lse:latest .
```

### Sandbox target (agent workspace, optional)

The same file has a second target, `sandbox`, for working *inside* a box rather
than deploying one — a Vercel Sandbox, a remote dev container, anything where a
person or an agent edits the checkout. It is built on the `runtime` stage, so it
runs the image's interpreter and its hash-pinned lock, and adds only what a
workspace needs:

```dockerfile
FROM runtime AS sandbox
USER root                                        # a clone must be writable
RUN apt-get install -y git curl ripgrep less procps
RUN apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils \
                       ghostscript qpdf          # the OCR toolchain (below)
RUN pip install -r requirements-dev.txt          # mypy, boto3, pyyaml, otel
RUN pip install "ocrmypdf>=16.0"                 # requirements-local.txt's floor
COPY tests/ scripts/ deploy/ nginx/ examples/ ./
# ... plus every root page, .github/workflows, pyproject.toml, docker-compose.yml
RUN git init && git add -A && git commit -m "baseline"   # so `git status` is usable
ENV VA_LSE_HEALTH_HOST=127.0.0.1                 # see §7
CMD ["streamlit", "run", "run_app.py", "--server.address=0.0.0.0"]
```

Four things about it are deliberate and easy to get wrong by hand:

* **A custom image, not a managed one.** The managed sandbox images ship Python
  3.14, and this lock's hash-pinned set does not install there: resolution stops
  at `httptools==0.8.0`, whose cp314 wheel exists on PyPI but is not among the
  hashes the lock carries (measured with `pip install --dry-run
  --require-hashes --only-binary=:all: --python-version 3.14 --platform
  manylinux_2_39_x86_64 --implementation cp --abi cp314 -r requirements.lock`).
  `jiter==0.17.0`, which this note used to name as the first refusal, carries a
  cp314 wheel hash and resolves on 3.14 — so the pin named here has to be the one
  that actually refuses, or the next reader checks the wrong package.
* **No `--server.port`.** 8501 is already Streamlit's default, and a *set* port is
  fatal when Streamlit resolves as a development-layout install —
  `server.port does not work when global.developmentMode is true` — which is what
  a `pip install -e` or a vendored Streamlit in the box would be. See
  `.streamlit/config.toml`.
* **`VA_LSE_HEALTH_HOST=127.0.0.1`.** Sandbox ports are published as public URLs
  and the sidecar's routes carry no authentication (§7), so the image keeps it on
  loopback; publish `8001` *and* override this only if you mean to.
* **OCR tooling, and only here.** The app deliberately has no OCR dependency and
  never shells out (`scripts/ocr_records.py`), so a page that is a scan is counted
  and reported, never read — records from a portal are routinely half scans, so
  that is a real gap, and the box is where it closes. `scripts/ocr_and_extract.py`
  is the entrypoint: it OCRs every image-only page in a bundle, then extracts with
  the app's own reader **under the original file names** (citations have to point
  at the record the user has, not at `.ocr.pdf`) and writes the queue's document
  JSON for the app to consume:

  ```bash
  python scripts/ocr_and_extract.py /work/records --out /work/bundle.json
  # --report-only        say which pages need OCR, change nothing
  # --no-ocr             reproduce what the app sees today (all scans unreadable)
  ```

  Exit codes: `0` extracted, `1` no record files found, `2` scans present and no
  OCR tooling installed, `3` bad input or nothing extractable. Nothing here is
  installed in the `runtime` stage — `tests/test_sandbox_image.py` fails if the
  deployment image grows a PDF renderer or an OCR engine.
* **The app can read records on the box instead (a swap, not a rewrite).**
  `app/extractors.py` implements the same `RecordExtractor` port as the reader in
  `app/documents.py`, so setting `VA_LSE_EXTRACTOR=sandbox` plus
  `VA_LSE_EXTRACTOR_RUNNER` moves *where* bytes are read without changing how text
  is shaped: the box runs the entrypoint above, this app maps the report JSON back
  through `app/job_payload.documents_from_json`, and a document answered under a
  name the user never uploaded is refused. Every failure — no runner, no tooling on
  the box, a refusal, a timeout (capped by the run's own remaining budget),
  unparseable JSON — is logged once through `app.error_report` and the file is read
  in-process, so a misconfigured box costs a warning rather than a run.

  Measured on a generated 20-file / 400-page all-scan bundle (`--no-ocr` vs. the
  box, engine cost excluded): **0 → 20 documents read, ≈124,000 characters of record
  text (≈31,000 tokens) reaching the digest**, at ≈4 ms per page of non-engine work.
  On a mixed bundle (born-digital pages, scans, one part-digital file) the today-path
  silently carries 14 of 120 pages with no text and refuses 2 of 6 files; the box
  answers with text on every page.
* **Vercel Sandbox is one script away, and it ships here.**
  `scripts/vercel_sandbox_runner.py` is the `VA_LSE_EXTRACTOR_RUNNER` command for a
  Vercel box — one staged file in, the box's report JSON out — so a deployment turns
  the swap on with two variables:

  ```bash
  VA_LSE_EXTRACTOR=sandbox
  VA_LSE_EXTRACTOR_RUNNER="python scripts/vercel_sandbox_runner.py {work}"
  ```

  Use an interpreter that actually exists. The runner is spawned as a subprocess, so
  the command is only as good as its first word: macOS ships `python3` and no `python`,
  and Streamlit started from a virtualenv does not put `python` on `PATH` either — the
  file then fails open with "the runner command does not exist (python)" and every
  record is read in-process, which is a warning per run rather than an error. Point it
  at an absolute interpreter path (`…/venv/bin/python`) or at `python3`.

  Per file it runs `sandbox create --name va-lse-ocr-<id> --image va-lse-sandbox:latest
  --timeout 20m --non-persistent --silent`, then `exec … mkdir -p
  /work/bundle/<the label's directory>`, `copy <work>/<file> <name>:/work/bundle/<label>`,
  `exec … python3 /app/scripts/ocr_and_extract.py /work/bundle --out
  /work/report.json --force`, `copy <name>:/work/report.json <work>/report.json`, and
  `sandbox remove <name>` in a `finally`. Three details are the load-bearing ones:
  the report comes back as a **file**, because `app/extractors.py` parses the runner's
  whole stdout as that JSON and `sandbox exec`'s stdout carries more than the command's;
  the **label is copied to its own path** (`records/2024/visit note.pdf` lands at that
  path under `/work/bundle`, so the entrypoint answers under the name the citation
  needs, and a label that climbs out of the bundle is refused rather than rewritten);
  and the staged bytes are checked against the manifest's **sha256** before a box is
  booted.

  | Knob | Default | Notes |
  |---|---|---|
  | `VA_LSE_SANDBOX_CLI` | `sandbox` | Split like a shell command, so `sbx`, `npx sandbox` or a wrapper of your own works. Install with `npm i -g sandbox` |
  | `VA_LSE_SANDBOX_IMAGE` | `va-lse-sandbox:latest` | The VCR image from the block below. Set it to `none` to drop `--image` and boot the CLI's *default runtime* — the way to prove a credential and the create/exec/copy/remove cycle before anything is pushed (measured: 4.4.0's default runtime is Python 3.14, which this lock refuses, so `none` is a probe and not a way to read records). A `create` that answers 404 says so and names this option |
  | `VA_LSE_SANDBOX_TIMEOUT` | `20m` | Box session timeout. It is also the backstop for a box whose runner was SIGKILLed, so keep it above `VA_LSE_EXTRACTOR_TIMEOUT_SECONDS` — and raise both together for large bundles |
  | `VA_LSE_SANDBOX_SCOPE` / `VA_LSE_SANDBOX_PROJECT` | (empty) | Passed as `--scope` / `--project` on every call that takes them |
  | `VA_LSE_SANDBOX_TOKEN` / `VERCEL_AUTH_TOKEN` / `VERCEL_OIDC_TOKEN` / `VERCEL_TOKEN` | (empty) | Read in that order and passed as `--token`; unset everywhere means the CLI's stored `sandbox login` session. Only the first is this app's own name: `VERCEL_AUTH_TOKEN` is the variable the **CLI itself** reads (measured against 4.4.0 — exporting `VERCEL_TOKEN` does *not* authenticate it, it waits for an interactive login), `VERCEL_OIDC_TOKEN` is what a Function is handed, and `VERCEL_TOKEN` is the REST API's convention, which the CLI ignores — so the runner passes it explicitly. The value is never logged |

  Sandbox takes a Vercel **access token** (Account Settings → Tokens, scoped to the team)
  or the **OIDC token** a Function is handed — Vercel's recommendation, because nothing
  long-lived has to be stored. One Vercel credential is not on that list: an **AI Gateway
  API key** (`vck_…`) authenticates the LLM gateway, not compute
  (`COMPATIBILITY.md` → *Vercel credentials are not interchangeable*), and the runner
  refuses it in the token slot by name rather than letting the CLI answer 401.

  One prerequisite is not about this repository at all. **The project that owns the image
  needs a slug-safe name.** The registry path is
  `vcr.vercel.com/<team-slug>/<project-slug>/<repository>`, and Vercel slugifies the
  *team* (`Secondary_Condition_Finder` → `secondary-condition-finder`) but not a
  project: pushing from a project named `va_draft` uploads every layer and then fails
  with `NAME_INVALID: invalid project slug` (measured — the same request against
  `…/secondary-condition-finder/va_draft/…` answers `NAME_INVALID` while
  `…/va-lse-sandbox/…` answers `not_found`, which is a valid path with no tags in it
  yet). The image and the boxes therefore live in one dedicated, dash-safe project,
  which is what `VA_LSE_SANDBOX_PROJECT` names; the app's own Vercel project is
  unrelated to it.

  Two things to expect. **One microVM per file** — that is the port's shape (one file
  in, documents out) — and `--non-persistent` plus the `finally` mean each one leaves
  nothing behind. **A SIGKILLed runner cannot clean up**: the app's own per-file
  timeout kills the runner in a way it cannot catch, which is exactly why the box
  carries its own `--timeout`. A box nobody removed stops itself; an unreachable
  `sandbox remove` is a warning, not a failed file. SIGTERM — what a Cancel or a
  container stop sends first — does run the cleanup.

  `tests/test_vercel_sandbox_runner.py` drives all of it with the CLI faked:
  `tests/fake_sandbox_cli.py` stands in for the microVMs and runs the real entrypoint
  over the file the runner really copied in, so the manifest, the label-to-path
  mapping, the argv shapes, the copy-back, the JSON on stdout and the `finally` are
  asserted, and one test goes through `CommandBoxRunner` + `SandboxExtractor` so the
  contract the app parses is the contract the script prints.

  `tests/test_vercel_sandbox_live.py` is the other half — the CLI's own behavior, which
  no fake can pin. It needs a credential and skips without one, so CI is unaffected:

  ```bash
  VA_LSE_TEST_VERCEL_SANDBOX_TOKEN=vcp_... \
  VA_LSE_TEST_VERCEL_SANDBOX_SCOPE=<team> \
  VA_LSE_TEST_VERCEL_SANDBOX_PROJECT=<project> \
  VA_LSE_TEST_VERCEL_SANDBOX_CLI='npx -y sandbox' \
    python -m unittest tests.test_vercel_sandbox_live
  ```

  The knobs arrive under the `VA_LSE_TEST_*` prefix because `tests/hermetic.py` strips
  ambient `VA_LSE_*` configuration; the first test creates a box on the default
  runtime, copies a file in and back out and removes the box (the credential, the
  transport and the lifecycle, for a fraction of a cent and no registry write), and the
  second runs the whole runner so the entrypoint reads a real staged record — skipping,
  with the build command, while the image is not in the registry. Four assumptions in
  this section were corrected by running it against a real account: `remove` does take
  the auth flags, the CLI's credential variable is `VERCEL_AUTH_TOKEN`, a failed `create`
  reports a bare status rather than "Image not found" (the message stays in its response
  buffer, which is why the last output line is a `hint:`), and a failure inside `exec`
  still has to forward the box's output or the diagnosis is lost.

```bash
# Build and push it to Vercel Container Registry, then boot a sandbox from it
docker build --target sandbox -t va-lse-sandbox:latest .   # local smoke test
vercel vcr login docker             # docker needs its own login to vcr.vercel.com
vercel vcr build docker . va-lse-sandbox:latest --push
sandbox create --name va-lse-dev --image va-lse-sandbox:latest \
  --vcpus 4 --timeout 2h --publish-port 8501 --connect
```

Two things about the middle line. The stage has to be named, because `vercel vcr
build` runs the container tool and the Dockerfile's default target is the
deployment image — that is the `-- --target sandbox` the CI job passes. And
`VERCEL_TOKEN` authenticates the Vercel *CLI*, not docker: without a registry
login for the container tool the build succeeds and the push fails with "no basic
auth credentials". `vercel vcr login docker` mints a 12-hour project-scoped OIDC
credential for it; a long-lived token works instead, with the token as the
password and the **team ID** that owns the project as the username
(`--username <team ID> --password-stdin`).

`tests/test_sandbox_image.py` asserts this contract (the stage, root, the dev
extras, OCR tooling, git, the copied files) as *text*, and
`tests/test_ocr_and_extract.py` covers the entrypoint's decisions with the OCR engine
faked — the binaries are not installed in CI and must not be required. Two CI jobs
build the stage as a whole: **Build the sandbox image stage** runs `docker build
--target sandbox` on every event (nothing is pushed, no credentials are needed), and
on manual dispatch **Push the sandbox image and read a record on a real box** builds,
publishes it to Vercel Container Registry and then runs the live test below — which
is also how a machine without Docker gets an image. It is dispatch-only and skips
itself unless `VERCEL_TOKEN` is set, in the same shape as the live smoke job
(`VERCEL_SANDBOX_PROJECT` names the project that owns the registry repository,
`VERCEL_TEAM_ID` is the registry login's username, `VERCEL_SANDBOX_SCOPE` the team;
VCR creates the repository on the first push).

### Multi-stage variant (smaller image, optional)

```dockerfile
# Dockerfile.multistage
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.lock ./
RUN pip install --no-cache-dir --prefix=/install --require-hashes -r requirements.lock

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local
RUN apt-get update && \
    apt-get install -y --no-install-recommends zlib1g && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY app/ ./app/
COPY run_app.py ./
COPY scripts/ ./scripts/

RUN mkdir -p /app/logs && chown -R nobody:nogroup /app/logs
USER nobody

EXPOSE 8501 8001

CMD ["streamlit", "run", "run_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

## 7. Health probe wiring

Every instance exposes two probes via the `app/health.py` sidecar (default
port `8001`):

| Probe | Port | Path | Use | Status codes |
|---|---|---|---|---|
| **Liveness** | 8001 | `/health` | Is the process alive? | Always `200` once started |
| **Readiness** | 8001 | `/ready` | Can it serve traffic? | `200` = ready, `503` = not ready |

Both probes answer on the interface in `VA_LSE_HEALTH_HOST` (default `0.0.0.0`,
which is what a kubelet probe needs since it arrives from outside the pod). None
of the three routes authenticates — a kubelet cannot present a token — so where
that port is published to the internet rather than to a cluster network, set
`VA_LSE_HEALTH_HOST=127.0.0.1`: same-host probes and
`sandbox exec curl localhost:8001/health` keep working, a remote browser gets a
refused connection instead of your metrics.

**What makes `/ready` return `503`:**
- Missing `OPENAI_API_KEY`
- Missing `OPENAI_BASE_URL`
- LLM endpoint unreachable (1.4s timeout)
- Either `LLM_MODEL_MAIN` or `LLM_MODEL_FAST` not listed at `/models`
- Graceful shutdown in progress (`is_shutting_down()`)

### Kubernetes probe configuration

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8001
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

startupProbe:               # Prevents premature liveness kills during slow startup
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 10      # 10 × 5s = 50s max startup time
```

### Docker Compose health checks

```yaml
healthcheck:
  test: ["CMD", "wget", "-qO-", "http://localhost:8001/health"]
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 15s
```

---

## 8. Graceful shutdown at scale

When a pod receives SIGTERM (rolling update, scale-down, node drain):

1. **Signal handler fires** (`app/shutdown.py`) — sets `_shutdown_requested` event
2. **`/ready` flips to 503** — the load balancer stops routing new traffic
3. **New runs are rejected** — both Evaluate and Draft tabs show a warning
4. **Inflight runs finish** — up to `VA_LSE_SHUTDOWN_GRACE_SECONDS` (default 30s)
5. **Per-call timeout** — each LLM call is bounded by `VA_LSE_LLM_CALL_TIMEOUT_SECONDS`
   (default 300s) so a single hung call cannot block the drain forever
6. **SIGKILL arrives** — the orchestrator's `terminationGracePeriodSeconds` expires
   and the process is killed

### Kubernetes terminationGracePeriodSeconds

Set `terminationGracePeriodSeconds` in your pod spec to **at least**
`VA_LSE_SHUTDOWN_GRACE_SECONDS + 15` (buffer for signal delivery + cleanup):

```yaml
terminationGracePeriodSeconds: 45  # 30s grace + 15s buffer
```

### Rolling update strategy

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1       # One pod at a time (preserves capacity)
    maxSurge: 1             # One extra pod during rollout
```

With `maxUnavailable: 1`, Kubernetes:
1. Creates a new pod (readiness probe must pass before it gets traffic)
2. Sends SIGTERM to an old pod
3. Waits up to `terminationGracePeriodSeconds` for the old pod to drain
4. Repeats until all pods are updated

---

## 9. Environment variables reference

All deployment-relevant variables (see `README.md` for the full list):

| Variable | Purpose | Default | Recommended for production |
|---|---|---|---|
| `OPENAI_API_KEY` | LLM API key | (required) | Store in K8s Secret or Docker secret |
| `OPENAI_BASE_URL` | LLM endpoint | Perplexity Agent API (`https://api.perplexity.ai/v1`) | Verify against your provider; pin it explicitly in production rather than relying on a default |
| `LLM_MODEL_MAIN` | Analysis model | `perplexity/kimi-k3` | Match your plan |
| `LLM_MODEL_FAST` | Bulk digest model | `perplexity/glm-5.3-flash` | Match your plan |
| `VA_LSE_RECORDS_CONCURRENCY` | Parallel digest workers | `2` | Raise for higher-tier endpoints |
| `VA_LSE_MAX_CONCURRENT_LLM_CALLS` | Global LLM concurrency cap | `20` | Raise if running 100 users across N pods |
| `VA_LSE_HEALTH_PORT` | Health sidecar port | `8001` | Keep default; mount in Service |
| `VA_LSE_HEALTH_HOST` | Interface the sidecar binds to | `0.0.0.0` | Keep the default in a cluster (probes arrive from outside the pod). Set `127.0.0.1` wherever that port is published to the internet |
| `VA_LSE_SHUTDOWN_GRACE_SECONDS` | Drain timeout | `30` | 30–60 for large record sets |
| `VA_LSE_LLM_CALL_TIMEOUT_SECONDS` | Per-call timeout | `300` | 300–600 depending on endpoint speed |
| `VA_LSE_LOG_DIR` | Diagnostic log directory | (stdout only) | Set to `/app/logs` for persistent logs |
| `VA_LSE_AUDIT_LOG_DIR` | Audit log directory | `logs` | Set to `/app/logs` **on a PVC**, not an emptyDir |
| `VA_LSE_AUDIT_LOG_MAX_BYTES` | Audit file size before rotation | `10485760` | Sets the ceiling; see §16 for the retention interaction |
| `VA_LSE_AUDIT_LOG_BACKUPS` | Rotated audit files kept | `10` | Raise it if the backup interval could outrun rotation |
| `VA_LSE_AUDIT_RETENTION_DAYS` | Local age-based retention for rotated audit files | `7` | The live `audit.log` is never swept |
| `VA_LSE_AUDIT_ERROR_MESSAGES` | Record free-text `error_message` in audit entries | `1` | Set `0` when shipping audit logs off-pod |
| `VA_LSE_AUDIT_BACKUP_DESTINATION` | Off-pod audit backup backend | (empty = off) | `filesystem`, `s3`, `gcs`, or `azure` |
| `VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS` | Pass interval for `--loop` | `6` | Bounds how much a pod death can lose |
| `VA_LSE_AUDIT_BACKUP_CLOUD_RETENTION_DAYS` | Remote object retention | `90` | Prefer a bucket lifecycle rule over `--prune` |
| `VA_LSE_AUDIT_BACKUP_REQUIRED` | Fail the backup job when unconfigured | `0` | `1` in the shipped CronJob |
| `VA_LSE_RUN_LOG_MAX_BYTES` | Run-log size before rotation | `10485760` | Previously unbounded — see §16 |
| `VA_LSE_RUN_LOG_BACKUPS` | Rotated run logs kept | `5` | — |
| `VA_LSE_DISK_MIN_FREE_BYTES` | Log-volume free-space floor | `268435456` | `/health → disk.below_floor` when crossed |
| `VA_LSE_LOG_JSON` | JSON log format | `0` | `1` for ELK/CloudWatch/Datadog |
| `VA_LSE_JOB_QUEUE` | Submit runs to a worker pool (Pattern C) | `0` | `1` on the web tier and every worker |
| `VA_LSE_REDIS_URL` | Redis backend for the job queue | (empty) | `redis://va-lse-redis:6379/0` for Pattern C |
| `VA_LSE_JOB_QUEUE_TTL_SECONDS` | Lifetime of a finished job's payload/result | `86400` | Lower to reclaim Redis sooner |
| `VA_LSE_JOB_QUEUE_LEASE_SECONDS` | Worker heartbeat window before a job is re-queued | `900` | Must exceed the longest gap between progress updates |
| `VA_LSE_WORKER_CONCURRENCY` | Jobs one worker process runs at once | `1` | Raise with the worker pod's memory limit |
| `VA_LSE_WORKER_HEALTH_PORT` | Worker health sidecar port | `8002` | Keep default; probe `/health` on it |
| `VA_LSE_WORKER_ID` | Worker identity in logs and job records | hostname:pid | Set from `metadata.name`/pod name |
| `VA_LSE_JOB_QUEUE_INLINE_MAX_BYTES` | Job size above which documents go to the blob store | `262144` | Keep default unless Redis is oversized |
| `VA_LSE_BLOB_STORE` | Blob backend for job documents | `auto` | `filesystem` or `s3` to pin it explicitly |
| `VA_LSE_BLOB_DIR` | Filesystem blob root | `blobs` | `/app/blobs` on the shared RWX PVC |
| `VA_LSE_BLOB_S3_BUCKET` | S3-compatible bucket | (empty) | Only for `s3`; needs `requirements-s3.txt` |
| `VA_LSE_EXTRACTOR` | Where record text is read: `in-process` or `sandbox` | `in-process` | Leave `in-process` on worker pods; a box is an operator choice, and `sandbox` falls back to `in-process` per file |
| `VA_LSE_EXTRACTOR_RUNNER` | Command that runs `scripts/ocr_and_extract.py` in the box | (empty) | `{work}` is the staged directory; stdout must end with the report JSON. `python scripts/vercel_sandbox_runner.py {work}` drives a Vercel Sandbox |
| `VA_LSE_EXTRACTOR_TIMEOUT_SECONDS` | Ceiling for one file's box work | `900` | Also capped by the run's remaining budget (`VA_LSE_PIPELINE_TIMEOUT_SECONDS`) |
| `VA_LSE_SANDBOX_CLI` | The Sandbox CLI `scripts/vercel_sandbox_runner.py` invokes | `sandbox` | Split like a shell command (`sbx`, `npx sandbox`, a wrapper); needs `npm i -g sandbox` |
| `VA_LSE_SANDBOX_IMAGE` | VCR image one file's box boots from | `va-lse-sandbox:latest` | The image `vercel vcr build docker …` pushes (see §6) |
| `VA_LSE_SANDBOX_TIMEOUT` | Box session timeout, and the backstop when a killed runner cannot remove the box | `20m` | Keep above `VA_LSE_EXTRACTOR_TIMEOUT_SECONDS` |
| `VA_LSE_SANDBOX_SCOPE` | Team scope passed to every Sandbox CLI call | (empty) | Only when the token spans teams |
| `VA_LSE_SANDBOX_PROJECT` | Vercel project the sandbox belongs to | (empty) | Only when it is not the token's default project |
| `VA_LSE_SANDBOX_TOKEN` | Sandbox credential, read before the Vercel-named ones below | (empty) | Use it when the token should not live in a `VERCEL_*` variable |
| `VERCEL_OIDC_TOKEN` | Sandbox credential a Function is provisioned with automatically | (empty) | Vercel's recommendation: nothing long-lived to store |
| `VERCEL_TOKEN` | Sandbox credential passed to the CLI as `--token` | (empty) | A team-scoped **access token** (Account Settings → Tokens) — never an AI Gateway key; unset means the CLI's stored `sandbox login` |
| `VA_LSE_TRACING` | Emit OpenTelemetry traces | `0` | `1` on the web tier **and** every worker; needs `requirements-otel.txt` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector/APM intake for spans | `http://localhost:4318` | In-cluster collector Service, or a vendor OTLP endpoint |
| `VA_LSE_TRACE_SAMPLE_RATIO` | Fraction of runs traced | `1.0` | Lower it if the backend meters per span |
| `VA_LSE_TRACE_CHUNK_SPANS` | One span per record-digest chunk | `0` | Leave off: hundreds of spans per large run |
| `VA_LSE_TRACE_LLM_CALLS` | One span per LLM provider call | `0` | Leave off unless you need per-call endpoint latency |

### Secrets management in production

**Never mount `.env` as a file in containers.** Use one of:

1. **Kubernetes Secrets:**
   ```bash
   kubectl create secret generic va-lse-env --from-env-file=.env
   ```
   Reference in deployment:
   ```yaml
   envFrom:
     - secretRef:
         name: va-lse-env
   ```

2. **Docker secrets:**
   ```bash
   docker secret create va_lse_env .env
   ```
   Reference in compose:
   ```yaml
   secrets:
     - va_lse_env
   ```

3. **Cloud secret managers** (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager):
   Use the platform's CSI driver or init container to inject secrets as env vars.

4. **Platform secret stores with no env injection** (Streamlit Community Cloud): the
dashboard's **Settings → Secrets** editor writes `.streamlit/secrets.toml` inside the
deployment, and the app reads it as a fallback after the environment. This is the only
channel available there, because `.env` is git-ignored and never ships. See
[§14 Pattern D](#14-pattern-d--streamlit-community-cloud).

---

## 10. Scaling guidance

### How many pods?

Each pod runs one Streamlit process with one health sidecar. A single pod
handles **one concurrent evaluation/draft run** comfortably (the LLM client
has its own concurrency limiter and circuit breaker). For N concurrent users:

| Concurrent users | Recommended pods | `VA_LSE_MAX_CONCURRENT_LLM_CALLS` per pod |
|---|---|---|
| 1–5 | 1–2 | 20 |
| 5–20 | 3–5 | 20 |
| 20–50 | 5–10 | 20 |
| 50–100 | 10–20 | 20 |

### When to autoscale

Use Kubernetes HPA (Horizontal Pod Autoscaler) with CPU-based scaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: va-lse
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: va-lse
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### LLM concurrency across pods

Each pod's `VA_LSE_MAX_CONCURRENT_LLM_CALLS=20` is **per-pod**. With 10 pods,
the total concurrent LLM calls could reach 200. Ensure your LLM endpoint can
handle this. Reduce `VA_LSE_MAX_CONCURRENT_LLM_CALLS` per pod if your provider
has a global rate limit:

```bash
# Example: 10 pods × 5 calls each = 50 max concurrent
VA_LSE_MAX_CONCURRENT_LLM_CALLS=5
```

### Circuit breaker interaction

The circuit breaker is **per-process** (not shared across pods). Each pod
independently detects LLM degradation and opens its own breaker. This is correct —
a pod that cannot reach the LLM should fail fast on its own, not affect other pods.

### Shared state across pods

| State | Stored in | Survives pod restart? | Shared across pods? |
|---|---|---|---|
| `st.session_state` | In-memory (bound to the WebSocket) | No — in every pattern | No — the live session cannot move pods |
| `request_id` | `st.session_state` + `ContextVar`, echoed into the job payload | No | No |
| Queued jobs + results | Redis (`va_lse:job:*`) | Yes | Yes |
| Job progress / status | Redis (`va_lse:job:*:meta`) | Yes | Yes |
| Audit logs | `logs/audit.log` (PV) | Yes | Yes (if shared PVC) |
| Diagnostic logs | `logs/app.log` (PV) | Yes | Yes (if shared PVC) |
| `usage_history.json` | Filesystem | No | No (per-pod) |
| Circuit breaker | In-memory | No | No (per-pod is correct) |
| Health cache | In-memory | No | No (per-pod is correct) |

Note the first row: **`st.session_state` is never shared in any pattern.** A run's inputs
and results move between pods through the job queue, which is why a project may render on a
different pod than the one that submitted it — while the browser session itself stays
pinned to one pod via affinity.

---

## 11. Rate limiting (reverse proxy)

Streamlit has no built-in HTTP rate limiting. For production deployments,
protect the app from request floods with rate limiting at the reverse proxy
layer. The app's per-request work (file upload, LLM call) is expensive — a
flood of concurrent Evaluate/Draft runs can exhaust the LLM endpoint and
cascade to all users.

### nginx rate limiting

nginx's `limit_req` module enforces request-rate limits per client IP.
Add these directives to the `http` block in `nginx/nginx.conf`:

```nginx
http {
    # --- Rate limiting zones ---
    # Zone 1: General requests (page loads, static assets).
    # 100 requests/minute per client IP, burst of 20.
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/m;

    # Zone 2: Action endpoints (Evaluate/Draft runs).
    # 10 requests/minute per client IP — these are expensive LLM calls.
    limit_req_zone $binary_remote_addr zone=actions:10m rate=10r/m;

    # Zone 3: Upload endpoints (file ingestion).
    # 20 requests/minute — uploads are CPU-bound (PDF extraction).
    limit_req_zone $binary_remote_addr zone=uploads:10m rate=20r/m;

    # Custom error page for rate-limited requests.
    limit_req_status 429;

    upstream streamlit_backends {
        ip_hash;
        server streamlit-web-1:8501;
        server streamlit-web-2:8501;
        server streamlit-web-3:8501;
    }

    server {
        listen 80;
        server_name _;

        # Health probes — never rate-limited.
        location /nginx-health {
            access_log off;
            return 200 '{"status":"ok","service":"nginx"}';
            add_header Content-Type application/json;
        }

        location /health {
            limit_req zone=general burst=5 nodelay;
            proxy_pass http://streamlit_backends;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /ready {
            limit_req zone=general burst=5 nodelay;
            proxy_pass http://streamlit_backends;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        # --- Main Streamlit traffic ---
        location / {
            limit_req zone=general burst=20 nodelay;
            proxy_pass http://streamlit_backends;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
            proxy_buffering off;
            client_max_body_size 60m;
        }

        # --- Upload endpoint (if exposed separately) ---
        # Streamlit handles uploads on the main path, but if you add a
        # dedicated upload route, apply the stricter upload zone:
        # location /upload {
        #     limit_req zone=uploads burst=5 nodelay;
        #     proxy_pass http://streamlit_backends;
        #     proxy_http_version 1.1;
        #     client_max_body_size 60m;
        # }
    }
}
```

**Key parameters:**

| Zone | Rate | Burst | Why |
|---|---|---|---|
| `general` | 100r/m per IP | 20 | Page loads, static assets, health probes |
| `actions` | 10r/m per IP | 5 | Evaluate/Draft runs — each triggers multiple LLM calls |
| `uploads` | 20r/m per IP | 5 | File uploads — CPU-bound PDF extraction |

**Rate-limit response:** nginx returns `429 Too Many Requests` with the
`Retry-After` header. Streamlit's frontend will show an error; users should
wait and retry. For a friendlier UX, add a custom error page:

```nginx
error_page 429 = @rate_limited;
location @rate_limited {
    default_type application/json;
    return 429 '{"error": "rate_limited", "message": "Too many requests. Please wait a moment and try again."}';
}
```

### Cloudflare rate limiting

If you use Cloudflare as a reverse proxy, configure rate limiting via
**Security → WAF → Rate limiting rules**:

| Rule | Expression | Rate | Action |
|---|---|---|---|
| **General flood** | `http.request.uri.path eq "/"` | 100 req/min per IP | Challenge (CAPTCHA) |
| **Action flood** | `http.request.uri.path eq "/"` AND `http.request.method eq "POST"` | 10 req/min per IP | Block |
| **Upload flood** | `http.request.uri.path contains "upload"` OR `http.request.uri.path eq "/"` | 20 req/min per IP | Challenge |
| **Health probe** | `http.request.uri.path in {"/health" "/ready"}` | 300 req/min per IP | Allow (never block health) |

**Cloudflare Page Rules** (legacy, simpler):

```
*va-lse.example.com/*
  → Rate Limiting: 100 requests per minute per IP
  → Action: Challenge
```

### Per-user session limits

The circuit breaker and concurrency limiter (`app/circuit_breaker.py`) cap
global LLM calls per pod. Per-*session* serialization comes from two mechanisms:

| Mechanism | Pattern | How |
|---|---|---|
| The script run owns the session | A/B, C | A running pipeline blocks the Streamlit script run, so a second click in that session is not processed until it returns. |
| Pending-job guard | C only | If the UI stops waiting on a queued job, the job id stays in `st.session_state`; `app/views/job_runner.py:submit_job` refuses a second submission until that job reaches a terminal state. Without it a user could queue a duplicate digest over the same records after giving up on a slow run. |
| Shutdown gate | A/B, C | `app/shutdown.py` rejects new runs once SIGTERM has been received. |

For additional server-side enforcement:

| Limit | Mechanism | Default |
|---|---|---|
| Concurrent runs per session | Streamlit button disable + `enter_run()` gate | 1 Evaluate + 1 Draft |
| Concurrent LLM calls per pod | `VA_LSE_MAX_CONCURRENT_LLM_CALLS` semaphore | 20 |
| Queue depth before rejection | `VA_LSE_LLM_QUEUE_MAX_DEPTH` | 50 |
| Circuit breaker failure threshold | `VA_LSE_CB_FAILURE_THRESHOLD` | 3 consecutive failures |

### Monitoring rate-limit rejections

Monitor nginx access logs for `429` responses. Alert if the rejection
rate exceeds 5% of total traffic:

```bash
# Count 429s in the last 5 minutes
tail -n 10000 /var/log/nginx/access.log | \
  awk -v cutoff="$(date -d '5 minutes ago' '+%d/%b/%Y:%H:%M')" '$4 > "["cutoff' | \
  grep '" 429 ' | wc -l

# Or with Prometheus + nginx-exporter:
# rate(nginx_http_requests_total{status="429"}[5m])
#   / rate(nginx_http_requests_total[5m]) > 0.05
```

**Recommended alerts:**
- Warning: 429 rate > 2% of traffic for 5 minutes
- Critical: 429 rate > 5% of traffic for 2 minutes
- Info: any single IP hitting 50+ requests/minute (potential abuse)

### Rate-limiting checklist

- [ ] nginx `limit_req_zone` configured for general, action, and upload zones
- [ ] Health probes (`/health`, `/ready`) are exempt from rate limiting
- [ ] `429` response includes `Retry-After` header or custom error page
- [ ] Cloudflare rate-limiting rules mirror nginx config (if using Cloudflare)
- [ ] Per-user session limits enforced by Streamlit button disable + `enter_run()`
- [ ] Monitoring dashboard shows 429 rate, top offending IPs, and trends
- [ ] Alert configured for >5% rejection rate

---

## 12. TLS and reverse proxy

Streamlit cannot set arbitrary HTTP response headers. For production,
always front the app with a TLS-terminating reverse proxy that adds:

```
Content-Security-Policy: default-src 'self' 'unsafe-inline' 'unsafe-eval'
X-Frame-Options: SAMEORIGIN
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
```

### Cloudflare / AWS ALB / GCP LB

All three provide TLS termination + health check integration:
- Point their health checks at `GET /health` (liveness) and `GET /ready` (readiness)
- Configure the load balancer's idle timeout to ≥ 600s (LLM calls can take minutes)
- Enable WebSocket support (Streamlit uses WebSockets for live updates)

### nginx reverse proxy (standalone)

```nginx
# nginx-tls.conf
server {
    listen 443 ssl http2;
    server_name va-lse.example.com;

    ssl_certificate /etc/ssl/certs/va-lse.crt;
    ssl_certificate_key /etc/ssl/private/va-lse.key;

    add_header Content-Security-Policy "default-src 'self' 'unsafe-inline' 'unsafe-eval'" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location / {
        proxy_pass http://streamlit_backends;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 600s;
        proxy_buffering off;
    }
}

server {
    listen 80;
    server_name va-lse.example.com;
    return 301 https://$host$request_uri;
}
```

---

## 13. Distributed cache for VA reference data

For multi-instance deployments, VA reference data (condition topics, rating
tables) benefits from a shared cache so every pod serves the same data without
redundant upstream fetches. The app uses `app/shared_cache.py` — a two-tier
cache with Upstash Redis as the shared tier and a process-local LRU as fallback.

### Quick setup (Upstash free tier)

```bash
# 1. Create a Redis database at https://console.upstash.com/
# 2. Copy the REST URL and token (HTTP API, not the redis:// URL)
# 3. Add to .env:
VA_LSE_SHARED_CACHE_URL=https://your-db.upstash.io
VA_LSE_SHARED_CACHE_TOKEN=AYxx...

# 4. Verify after deploy:
curl -s http://localhost:8001/health | python -c "import sys,json; d=json.load(sys.stdin); print(d['cache'])"
# Should show: {backend: 'upstash_redis', is_shared: true, reachable: true, hit_rate: 0.0, ...}
```

### K8s deployment

```bash
# Add to your K8s secret:
kubectl patch secret va-lse-env -p \
  '{"stringData":{"VA_LSE_SHARED_CACHE_URL":"https://your-db.upstash.io","VA_LSE_SHARED_CACHE_TOKEN":"AYxx..."}}'
```

### Docker Compose deployment

Add to `docker-compose.yml` environment section:
```yaml
environment:
  - VA_LSE_SHARED_CACHE_URL=https://your-db.upstash.io
  - VA_LSE_SHARED_CACHE_TOKEN=${VA_LSE_SHARED_CACHE_TOKEN}
```

### Monitoring

The health endpoint (`GET /health`) includes a `cache` field with:
- `backend`: `upstash_redis` or `local_lru`
- `is_shared`: `true`/`false`
- `reachable`: `true`/`false` (for Upstash)
- `hits`, `misses`, `hit_rate`, `errors` — effective cache utilization

It also includes a `job_queue` field (see §4, Pattern C):
- `backend`: `redis`, `upstash_rest`, or `inprocess`
- `is_distributed`: `true` only when a worker in another process can claim work
- `enabled`: whether `VA_LSE_JOB_QUEUE=1`
- `depth`: jobs waiting to be claimed

Alert on `depth` growing without bound (workers dead or under-provisioned), and on
`is_distributed: false` while `enabled: true` — that combination means runs are
being queued into a backend no worker can reach.

---

## 14. Pattern D — Streamlit Community Cloud

Streamlit Community Cloud runs this repository directly: no container, no orchestrator, and
**no `.env`** — the file is git-ignored, so it never reaches the deployment. Configuration
arrives through the platform's secrets manager, which the app reads as a fallback after the
process environment.

### Configure the deployment

In the Streamlit Cloud dashboard: your app → **⋮ → Settings → Secrets**. This writes
`.streamlit/secrets.toml` inside the running deployment, using the same value names as the
environment variables above. Add your provider key there under `OPENAI_API_KEY`, and the
endpoint and models as:

```toml
OPENAI_BASE_URL = "https://api.perplexity.ai/v1"
LLM_MODEL_MAIN = "perplexity/kimi-k3"
LLM_MODEL_FAST = "perplexity/glm-5.3-flash"
```

(Pin the endpoint and models explicitly in a deployed environment. The code defaults are the
same values, but a pin is what makes the deployment's behaviour independent of an app upgrade.)

Optional extras, same file: `FETCH_SANDBOX_API_KEY`, `FETCH_SANDBOX_BASE_URL`,
`FETCH_SANDBOX_RECORDS_PATH`, `VA_LSE_SHARED_CACHE_URL`, `VA_LSE_SHARED_CACHE_TOKEN`.

Then **reboot the app** — the running process reads its configuration at startup, so secret
edits do not affect it until it restarts.

### Dependencies on a host with no shell

Community Cloud installs **`requirements.txt`** from the repository when it builds the app.
There is no shell on the platform, so a package that is not in that file cannot be added at
runtime — a deploy-time gap that surfaces as a *feature* politely refusing to run, never as an
error anyone can act on from the dashboard.

That is why the Perplexity SDK is listed in the **core** `requirements.txt` even though the
integration treats it as optional in code: with it as an optional `requirements-*.txt`, the
Research tab and the framework-currency check could only ever render "install this package"
here. Nothing else needs it, and every import of it stays function-local, so a partial install
elsewhere still degrades to an explanatory message rather than an `ImportError`.

Two consequences for a hosted deployment:

* **Docker, CI, and K8s use `requirements.lock`** (`pip install --require-hashes -r
  requirements.lock`, §5), so the SDK must be in both files — keep them in step with
  `pip-compile` (see `README.md → Dependency locking`). Community Cloud ignores the lockfile
  and resolves `requirements.txt` unpinned, so a deployed upgrade can pick up a newer SDK than
  CI tested.
* **The key is still required.** The package ships the *ability* to research; the features
  stay off until `PERPLEXITY_API_KEY` is set (auto-derived from `OPENAI_API_KEY` when the base
  URL is Perplexity's, see the resolution order below). Reboot after adding it.

### Resolution order

1. process environment (platform-injected var, K8s/Docker secret, CI secret),
2. the project `.env` (local development only),
3. Streamlit secrets (`.streamlit/secrets.toml`),
4. the code defaults in `app/config.py`.

Environment wins over secrets, so a local `.env` always overrides a hosted secret store. When a
value comes from the secrets manager the sidebar captions it (`🔐 From Streamlit secrets: …`),
so a pre-filled field is never mistaken for one the user typed.

### Key → endpoint pairing is the most common hosted failure

A key is only valid against the endpoint that issued it: a key issued for one plan family is
rejected by the other's gateway, and vice versa. When the pairing is wrong the gateway returns
a rejected-key error that looks nothing like "wrong host" — and because nothing but the auth
check runs, the UI reports it almost instantly, which reads as a silent failure.

Set the key and the base URL from the same provider account, then click **Test connection** in
the sidebar: it runs the same preflight a run does on the on-screen values — the model listing
plus one real call per configured model — and should answer that a real call answered, so the
configured key, endpoint and ids all work.

Remember **Apply settings**: the API key is read live on every run, but the base URL and model
names only take effect after clicking it (the sidebar warns while a change is pending), and
`GET {base_url}/models` is the same check the app performs at startup.

### Hosted checklist

- [ ] `OPENAI_API_KEY` + `OPENAI_BASE_URL` set together in **Settings → Secrets** (same account)
- [ ] `LLM_MODEL_MAIN` / `LLM_MODEL_FAST` name models the endpoint actually serves
- [ ] Perplexity SDK present in `requirements.txt` (makes the Research tab and the
      framework-currency check loadable on the platform); no `pip install` step is possible here
- [ ] `PERPLEXITY_API_KEY` set if the base URL is *not* Perplexity's, or the Research tab is off
- [ ] App rebooted after editing secrets
- [ ] Sidebar **Test connection** reports a real call answered on the deployed URL
- [ ] Sidebar captions the values as `🔐 From Streamlit secrets:`
- [ ] No key typed into a chat, issue, or commit; rotate anything that was exposed
- [ ] Long runs: leave the tab open — a real Evaluate takes minutes
- [ ] Upload sizes fit `.streamlit/config.toml` (`maxUploadSize = 50` MB per file)

### Limits to plan around

- **One process serves every user.** `VA_LSE_MAX_CONCURRENT_LLM_CALLS` (default 20) and the
  circuit breaker still apply, but there is no per-pod scaling here — set
  `VA_LSE_SHARED_CACHE_URL/TOKEN` to keep upstream reference fetches shared rather than repeated.
- **No persistent disk.** `logs/audit.log`, `logs/runs.jsonl`, and `outputs/` live inside the
  container and disappear on restart/reboot. The About tab's run log is the record you have; if
  you need a durable audit trail, deploy with Pattern A/B/C and point `VA_LSE_AUDIT_LOG_DIR` at
  a volume.
- **Secrets are per-deployment, not per-user.** Every visitor shares the configured key, so
  plan quota accordingly, and treat the deployment URL as authorized-user-only.

---

## 15. Distributed tracing (OpenTelemetry)

Logs and the profiler already answer "what happened on this pod". Tracing adds the two
things they cannot: a **span tree per run** (which phase was slow, for *this* run, with
real start/end times) and a trace that **survives the process boundary** — in Pattern C the
digest runs on a worker, so a log-derived view of a run stops at the queue.

Off by default, and a no-op when the packages are absent, so nothing changes for a
deployment that does not want it. Full reference: [`TRACING.md`](TRACING.md).

```bash
pip install -r requirements-otel.txt

kubectl patch secret va-lse-env -p \
  '{"data":{"VA_LSE_TRACING":"MQ==","OTEL_EXPORTER_OTLP_ENDPOINT":"aHR0cDovL290ZWwtY29sbGVjdG9yLm9ic2VydmFiaWxpdHk6NDMxOA=="}}'

kubectl rollout restart deployment/va-lse deployment/va-lse-worker
curl -s localhost:8001/health | jq .tracing     # web pod
curl -s localhost:8002/health | jq .tracing     # worker pod  ← both must show active: true
```

### Backends

The exporter is OTLP over HTTP, so Jaeger, Grafana Tempo, an OpenTelemetry Collector,
Datadog, New Relic and Honeycomb all work through the same two variables — switching
backends is an env change, not a code change (recipes per vendor are in `TRACING.md`).
For a self-hosted stack:

```bash
# Collector (otlp receiver → your storage), then point the app at its Service
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
```

### What the trace looks like in Pattern C

```
queue:submit (web pod)
└── run:evaluate (worker pod)
    ├── records:review
    │   └── records:digest          chunks, concurrency, pages
    ├── claims / verify / rubric / topic / revision / report
    └── …
```

The web pod injects its W3C trace context into the job payload, so the worker's spans join
the submit span rather than starting an unrelated trace. Both tiers therefore need
`VA_LSE_TRACING=1` and the same collector endpoint; a worker without it traces nothing, and
a web pod without it makes the worker start a fresh trace per job.

### Sampling and cost

Runs are long and few, so the default (`VA_LSE_TRACE_SAMPLE_RATIO=1.0`) traces everything:
a 2,000-page Evaluate is roughly a dozen spans. Sampling is **parent-based**, so a sampled
run keeps its whole tree and a worker inherits the web pod's decision. Two knobs are off by
default for cardinality reasons — `VA_LSE_TRACE_CHUNK_SPANS` and `VA_LSE_TRACE_LLM_CALLS`
add one span per chunk or per LLM call, which on a 5,000-page set is thousands of spans per
run. Turn them on deliberately, ideally with a partial sample ratio.

### PHI and data egress

Traces are metadata only: phase names, counts, page/char sizes, model names, job ids, error
classes, and the run's `request_id`. Statement text, observations, record text and prompts
are excluded, and attribute names that look like free text (including any `*_text`) are
dropped before the SDK sees them — there is a test asserting that
(`tests/test_tracing.py → TestPiiScreening`).

**Spans do leave the deployment.** If the deployment is PHI-sensitive, run a self-hosted
collector inside your network (Jaeger/Tempo/OTel Collector) rather than a third-party SaaS;
picking the backend is a compliance decision as much as an operational one.

### Behaviour under failure

- A collector that is down or misconfigured never fails a run — the exporter logs a warning
  and drops batches. `health.tracing.active` reports configuration, not reachability, because
  `/health` is polled by probes and must not block on a network round trip.
- Spans are buffered and flushed on graceful shutdown, **after** the drain
  ([§8](#8-graceful-shutdown-at-scale)): the trace of the run that just finished is exported
  instead of dying with the process. `python -m app.worker --once` flushes on exit too.
- Tracing adds no secrets to the pod: the OTLP headers are read from the same env/secret
  mechanism as everything else.

```bash
# Confirm from a shell that tracing is off/on and why
curl -s localhost:8001/health | jq '.tracing | {enabled, active, exporter, endpoint, reason}'
```

### Tracing checklist

- [ ] `pip install -r requirements-otel.txt` in the image (or leave tracing disabled)
- [ ] `VA_LSE_TRACING=1` on the web tier **and** on every worker pod
- [ ] `OTEL_EXPORTER_OTLP_ENDPOINT` points at a collector reachable from both tiers
- [ ] `GET /health → tracing.active` is true on both tiers
- [ ] A test run appears in the backend as one trace spanning the web pod and the worker
- [ ] Sampling ratio chosen for your backend's pricing
- [ ] Collector is self-hosted if the deployment is PHI-sensitive

---

## 16. Audit log retention and backup

The audit stream (`logs/audit.log`) is the forensic record of every Evaluate/Draft:
which action ran, when, with what inputs (counts and classifications only), and how it
ended. It is written by [`app/audit.py`](app/audit.py) as a JSON-lines stream separate
from the diagnostic `app.log` so it can be retained under a different policy. Two gaps
made it insufficient as a compliance artifact, and both are load-bearing:

1. **It lived only on the pod.** In the original manifests `/app/logs` was an
   `emptyDir` on the web Deployment *and* on the worker. A restart reclaimed the
   stream, and in Pattern C the two tiers kept *separate* `audit.log` files on two
   ephemeral disks — so "the audit log" was really two partial files that vanished
   together. **No backup strategy can fix this; the volume has to change first.**
2. **Rotation bounds size, not time.** `VA_LSE_AUDIT_LOG_MAX_BYTES` ×
   (`VA_LSE_AUDIT_LOG_BACKUPS` + 1) is a hard ~110 MiB ceiling, and the rotation
   handler deletes the oldest file to make room. There was no age rule at all, so a
   busy day could destroy a file that was minutes old.

### Step 1 — make the log volume persistent and shared

```bash
kubectl apply -f deploy/k8s/k8s-logs.yaml      # ReadWriteMany PVC: va-lse-logs
kubectl get pvc -n va-lse va-lse-logs          # must reach Bound, not Pending
```

`k8s-deployment.yaml` and `k8s-worker.yaml` already mount `va-lse-logs` at
`/app/logs`. `ReadWriteMany` is required because the backup job runs in its own pod —
check `kubectl get storageclass` first, since `local-path` and `gp2` are RWO-only.

The containers run as uid 65534 with a read-only root filesystem, so the volume must
be *writable by 65534*; the manifests set `fsGroup: 65534`, which is what most CSI
drivers need. A permission mismatch shows up as `audit.write_failures` climbing in
`/health`, not as a silent failure.

Docker Compose needs nothing here: `app-logs` is already a named volume shared by the
web tier, the workers, and the `audit-backup` service.

### Step 2 — choose a destination

| Destination | Set | Install | Notes |
|---|---|---|---|
| `filesystem` | `VA_LSE_AUDIT_BACKUP_DIR` | — | An NFS/Azure Files/EFS mount. A directory on the pod's *own* volume is permitted but reported `off_pod: false`, because it cannot survive the pod. |
| `s3` | `VA_LSE_AUDIT_BACKUP_S3_BUCKET` (+ `_PREFIX`, `_ENDPOINT_URL`) | `requirements-backup.txt` | AWS S3, Cloudflare R2, MinIO, DO Spaces. GCS also works in interoperability mode. |
| `gcs` | `VA_LSE_AUDIT_BACKUP_GCS_BUCKET` | `requirements-backup.txt` | Native GCS API. |
| `azure` | `VA_LSE_AUDIT_BACKUP_AZURE_CONTAINER` + `AZURE_STORAGE_CONNECTION_STRING` (or `_ACCOUNT_URL` + workload identity) | `requirements-backup.txt` | Azure Blob has no S3-compatible API, hence a native backend. |

Prefer workload identity (IRSA, GCP Workload Identity, Azure Workload Identity) over
static keys. The SDKs are imported lazily: a missing package is a clear configuration
error from the job, never an `ImportError` traceback.

### Step 3 — run the job

```bash
# Kubernetes (recommended shape): a CronJob in its own pod, every 6h
kubectl apply -f deploy/k8s/k8s-audit-backup.yaml
kubectl create job --from=cronjob/va-lse-audit-backup audit-backup-manual -n va-lse
kubectl logs -n va-lse job/audit-backup-manual

# Compose: the audit-backup service runs continuously (--loop --prune)
docker compose up -d audit-backup && docker compose logs -f audit-backup

# One-off / inspection, from anywhere with the log volume mounted
python scripts/backup_audit_logs.py --once --prune     # one pass, enforce retention
python scripts/backup_audit_logs.py --dry-run --once   # report, change nothing
python scripts/backup_audit_logs.py --status           # what /health would say
```

A **CronJob is preferred over a sidecar**: a sidecar shares its pod's lifecycle, so it
dies with the pod whose logs it is meant to rescue and cannot run during an eviction
or a rolling update. The CronJob uses `concurrencyPolicy: Forbid` plus
`activeDeadlineSeconds: 900`, and the script additionally takes a cross-process lock so
two backup processes cannot race the same watermark.

Exit codes are designed to be visible in `kubectl get jobs`: `0` success (or
unconfigured-and-not-required), `1` misconfiguration, `2` a pass that ran and failed.
The shipped CronJob passes `--require-destination` so an unconfigured job **fails**
rather than exiting 0 forever.

### What gets shipped, and why not just the rotated files

Each pass uploads rotated files *and* the un-shipped tail of the live `audit.log`,
cut back to the last complete line:

```
audit/2026/09/16/live-audit.log-g3a1b2c4d5e6f-0-48213-9f31c2ab7d10.jsonl
                └ file          └ file gen    └ byte range └ content hash
```

The `g<tag>` segment identifies **which** `audit.log` the range came from. Every
rotation starts a new file at offset 0, so without it a window from the file written
after a rotation is indistinguishable from a retry of a window from the file before
it — and a restore would concatenate the two files' heads and call the result
contiguous. Keys written by an earlier version (no `g` segment) still verify; they are
reported as generation `unknown` rather than merged into a neighbour.

Shipping only rotated files is the obvious implementation and it fails the actual
goal. At a few hundred runs a day a 10 MiB file takes *weeks* to rotate, so the live
file holds every recent event — including the run that just failed. The byte-range
watermark closes that gap without ever reading a half-written line, and the
content+range-addressed key means a pass that crashes after uploading but before
checkpointing **overwrites the same object on retry** instead of duplicating it.

Rotation is detected two ways, because either alone is insufficient: the inode
changes (`RotatingFileHandler` renames the old file and creates a new one) *and* a
size regression catches truncation. An inode check alone misses a same-size rewrite;
a size check alone misses a rotation where the new file happens to be the same length
as the watermark.

### Retention

| Rule | Setting | Default | Enforced by |
|---|---|---|---|
| Local size ceiling | `VA_LSE_AUDIT_LOG_MAX_BYTES` × (`_BACKUPS` + 1) | ~110 MiB | the rotation handler |
| Local age ceiling | `VA_LSE_AUDIT_RETENTION_DAYS` | 7 days | the backup pass (runs even with no destination) |
| Remote retention | `VA_LSE_AUDIT_BACKUP_CLOUD_RETENTION_DAYS` | 90 days | `--prune`, or a bucket lifecycle rule |

Effective local retention is **whichever ceiling is reached first** — a busy period
can delete a file well inside 7 days. That is the intended trade (a bounded volume
beats an unbounded one), but it is why the interval matters: set
`VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS` so a pass always runs before rotation can
consume `_BACKUPS` files. Lowering `VA_LSE_AUDIT_LOG_MAX_BYTES` or raising `_BACKUPS`
buys time if your audit volume is unusually high.

A **bucket lifecycle rule is the better instrument for remote retention** than
`--prune`: it survives a broken backup job. Use `--prune` only where the destination
cannot set one.

### Watch the disk, and the writes

```bash
curl -s localhost:8001/health | jq '{audit, audit_backup, disk}'
```

```jsonc
{
  "audit":        { "status": "ok", "write_failures": 0, "retention_days": 7 },
  "audit_backup": { "status": "ok", "destination": "s3", "last_success_utc": "…",
                    "pending_bytes": 0, "stale": false },
  "disk":         { "checked": true, "free_bytes": 528536432640, "below_floor": false },
  "restore":      { "restore_available": true, "destination": "s3", "off_pod": true,
                    "tool": "scripts/restore_audit_logs.py" }
}
```

- `audit.write_failures` is the one that matters most. `logging` swallows handler
exceptions and every audit call is best-effort, so a **full disk used to stop auditing
silently** while the app kept serving. It is now counted and reported as
`status: degraded` — the app is still up, but records are being lost.
- `audit_backup.status` is `ok` / `stale` / `error` / `never_ran` / `disabled`. It reads
the state file on the shared volume, never the network: the backup runs in a different
process, and `/health` must stay under its 2 s SLO. `stale` means one missed interval,
not a hard failure — audit logs are forensics, not real-time alerting.
- `pending_bytes` is the honest answer to "what would a pod death cost me right now?".

**`logs/runs.jsonl` was the actual disk-exhaustion risk.** It is a plain append with
no rotation, so unlike `audit.log` it could grow without bound on a long-lived pod.
Both are now bounded (`VA_LSE_RUN_LOG_MAX_BYTES`, `VA_LSE_RUN_LOG_BACKUPS`), and
`VA_LSE_DISK_MIN_FREE_BYTES` flags a volume approaching full *before* writes start
failing. Audit writes are never dropped to reclaim space — losing compliance records
to save bytes is the wrong trade, and it is now visible instead of silent.

### Verify the backup, and restore from it

A backup nobody has read back is a hypothesis. `scripts/restore_audit_logs.py` is the
companion to the backup job and answers the three questions an audit actually asks: is
the data intact, is any of it missing, and can the record be rebuilt.

```bash
# download and hash everything, then walk the byte ranges for holes
python scripts/restore_audit_logs.py --verify

# fast: skip the downloads (no integrity verdict, no gap analysis)
python scripts/restore_audit_logs.py --verify --no-hash

# rebuild the stream for an investigator
python scripts/restore_audit_logs.py --restore /tmp/audit-restore

# read from a different bucket than the deployment writes to (e.g. a read-only key)
python scripts/restore_audit_logs.py --verify --bucket audit-archive --prefix cold

# machine-readable, for a ticket or a compliance record
python scripts/restore_audit_logs.py --verify --json
```

Exit codes: `0` verified/restored, `1` misconfiguration (nothing configured), `2` the
backup is reachable but incomplete, corrupt, or unreachable. `--restore` exits `2`
when the stream it wrote has gaps — a holey restore must not look like a clean
success.

**Integrity needs no side manifest.** Every object key already ends in the first 12
hex characters of the SHA-256 of its own content, so the expected hash travels *with*
the data and cannot drift from it. A downloaded object whose bytes do not hash to the
value in its own key is reported `CORRUPT`. There is nothing to keep in sync, and
nothing to go stale.

**Gaps are the interesting part.** Windows are byte ranges, so an ordinary verify
reports the exact offsets that were never uploaded — the records written while the
backup job was failing:

```text
destination: s3 (off_pod=True)
objects: 41 (12,884,901 bytes)
hash-verified: 41 of 41
file generations: 3
  - audit.log gen=3a1b2c4d5e6f span 0-10485760 (7 window(s), holding 10582048 bytes)
  - audit.log gen=9f8e7d6c5b4a span 0-812 (1 window(s), holding 300 bytes)  [512 bytes missing inside]
  - audit.log gen=1a2b3c4d5e6f span 0-65536 (2 window(s), holding 65536 bytes)
GAPS (1) — 512 bytes never uploaded:
  - audit.log gen=9f8e7d6c5b4a offsets 300-812 (512 bytes)
VERDICT: attention needed
```

A gap means records are gone from both the pod and the destination. The remedy is not
in this tool — it is a shorter `VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS`, or a job that is
actually running.

**What a restore writes** (`--restore DIR`):

| File | Contents |
|---|---|
| `restored.jsonl` | The live stream, reassembled in upload order across generations — the chronological record |
| `rotated/<name>-<size>-<sha>.jsonl` | Every distinct rotated snapshot, named by content so two snapshots of one filename cannot collide |
| `manifest.json` | The full verification report: gaps, corrupt objects, line counts, what was included and skipped |

The rotated snapshots overlap `restored.jsonl` by design — a file is shipped live *and*
again in full once it rotates — and they are the only copy of any range the live
watermark never reached, so they are written out rather than folded in. If a gap is
reported **and** rotated snapshots are present, check the manifest before concluding
that a range is unrecoverable.

Two deliberate choices worth knowing:

- **The tool never writes to the destination.** Restoring is a read. A recovery tool
  that can delete objects during an incident is one that can destroy the only copy.
- **A corrupt object is still written, and flagged.** For forensics a flagged copy
  beats no copy; `manifest.json` and the exit code both say the object is damaged, so
  it cannot be mistaken for a verified record.

**One honest limitation: ordering between file generations.** Order *within* a file is
exact (byte offsets), but every generation is a fresh file numbered from 0, so the only
signal that distinguishes them is the upload time the destination reports. If a store
returns no modification times, or two generations collide, the chronological order is
unknowable — the verify reports `order_uncertain`, the exit code is `2`, and
`restored.jsonl` should be re-sorted on the `timestamp` field inside each record. Real
S3, GCS, Azure, and a filesystem all report times and generations are separated by
whole backup intervals, so this is the exception rather than the norm; it is reported
because a silently mis-ordered audit record has no other symptom.

Run `--verify` on a schedule of its own — monthly is usually right for a small business.
A backup job that reports success while writing objects nothing can read is the failure
mode this script exists to catch, and it is invisible from the backup job's own logs.

### Metrics for alerting (`GET /metrics`)

`/health` answers "is this instance alive"; it is the wrong instrument for "has the
backup been failing for three days". The health sidecar also serves Prometheus text
format on the same port, derived from the same payload — so every value is as cheap as
`/health` is, and a scrape cannot block on a slow dependency.

```yaml
# Prometheus scrape config
- job_name: va-lse
  metrics_path: /metrics
  static_configs:
    - targets: ["va-lse-web.va-lse.svc:8001"]
```

If you use the Prometheus operator, annotate the web and worker pods instead:
`prometheus.io/scrape: "true"`, `prometheus.io/port: "8001"`,
`prometheus.io/path: "/metrics"`.

| Metric | Meaning |
|---|---|
| `va_lse_audit_backup_state` | Enum: `0`=ok, `1`=disabled, `2`=never_ran, `3`=stale, `4`=error, `5`=unavailable |
| `va_lse_audit_backup_last_success_timestamp_seconds` | When the last pass succeeded |
| `va_lse_audit_backup_pending_bytes` | Audit bytes on this volume not yet shipped — what a pod death would lose |
| `va_lse_audit_backup_off_pod` | `0` means the destination shares the audit log's volume and cannot survive the pod |
| `va_lse_audit_write_failures_total` | Audit records lost to failed writes (usually a full or read-only volume) |
| `va_lse_disk_free_bytes`, `va_lse_disk_below_floor` | Headroom on the log volume against `VA_LSE_DISK_MIN_FREE_BYTES` |
| `va_lse_job_queue_depth`, `va_lse_job_queue_distributed` | Backlog, and whether a separate worker can claim jobs (Pattern C only) |
| `va_lse_tracing_enabled`, `va_lse_tracing_active` | Whether spans are configured and actually recording |
| `va_lse_llm_failover_enabled` | `1` when a second endpoint is configured — no failover exists at `0` |
| `va_lse_llm_failover_active` | `1` while calls are being served by the fallback endpoint |
| `va_lse_llm_primary_unhealthy_seconds` | How long the primary has been failing continuously (`0` while healthy) |
| `va_lse_llm_failover_after_seconds` | The configured grace period before failover engages |
| `va_lse_llm_failover_total` | Calls moved to the fallback because the primary failed |
| `va_lse_llm_endpoint_calls_total`, `va_lse_llm_endpoint_duration_ms` | Call volume and latency split by serving endpoint |
| `va_lse_circuit_breaker_state{breaker="llm-fallback"}` | The backup endpoint's own breaker (appears once it has been used) |

### Shipped monitoring assets

Rather than write these by hand, `deploy/monitoring/` contains a working stack for a
small-business deployment:

| File | What it is |
|---|---|
| `prometheus.yml` | Scrape config for the web and worker health ports, with the `/metrics` path |
| `alerts.yml` | The alert rules (availability, LLM, pipeline, compliance, self-monitoring) |
| `alerts.test.yml` | Unit tests for those rules — `promtool test rules alerts.test.yml` |
| `grafana-dashboard.json` | The operations dashboard (latency percentiles, breaker, queue, backup, failover) |
| `grafana-provisioning/` | Datasource + dashboard provisioning so Grafana loads it on start |
| `alertmanager.yml`, `blackbox.yml` | Example routing, and synthetic `/ready` probing |

Two conventions worth keeping if you edit them:

- **Every rule is tested.** `promtool check rules alerts.yml` validates the PromQL, and
  `promtool test rules alerts.test.yml` proves each rule fires *and* that it stays quiet
  in the healthy case. A rule that silently never matches is worse than no rule,
  because it is trusted. Re-run both after any edit.
- **A metric named in a dashboard or alert must exist.** `tests/test_monitoring_assets.py`
  fails if an asset references a `va_lse_*` name the app does not emit, and
  `promtool` catches the rest.

Alerting suggestions for a small-business deployment:

```promql
# the backup has stopped (2 = never ran, 3 = stale, 4 = error) — page
max_over_time(va_lse_audit_backup_state[6h]) > 1

# audit records are being dropped right now — page immediately
increase(va_lse_audit_write_failures_total[10m]) > 0

# the log volume is nearly full — ticket
va_lse_disk_below_floor == 1

# a destination on the pod's own volume is no protection — ticket
va_lse_audit_backup_off_pod == 0
```

For reference, the failover rules shipped in `alerts.yml`: `VaElseLlmRunningOnFallback`
(`va_lse_llm_failover_active == 1` for 5m — users are on the backup provider) and
`VaElseLlmPrimaryUnhealthy` (`va_lse_llm_primary_unhealthy_seconds > 120` for 5m — a
failure that precedes failover, or the whole story on a single-endpoint deployment).

**A value that could not be read is omitted, not zeroed.** If the queue backend is
remote, `va_lse_job_queue_depth` is absent rather than `0`, because a reported `0`
during an outage is worse than a missing series — one gets ignored, the other gets
alerted on. `/health` reports the same distinction as `depth: null` with
`depth_source: "not_probed"`. Use `/metrics?probe=1` (or the sidebar's **Check
backlog**) when you actually want that read performed, and expect it to be slower.

### PHI and data egress

Audit entries are designed to be non-PII (counts and classifications only), so backing
them up is lower-risk than exporting traces. One field is an exception and is worth
knowing about before you ship these logs to a third party: **`error_message` is
arbitrary upstream exception text** — a library can put anything in an exception
message, including fragments of what was sent to it. It is scrubbed of PII-shaped
tokens (SSNs, long digit runs, email addresses) and whitespace-collapsed, and
`VA_LSE_AUDIT_ERROR_MESSAGES=0` omits it entirely. For a cloud destination, set it to
`0`: `error_class` is still recorded and is always safe.

### Audit backup checklist

- [ ] `/app/logs` is a **PVC mounted by web, workers, and the backup job** — not an emptyDir
- [ ] `kubectl get pvc va-lse-logs` is `Bound`, and the volume is writable by uid 65534
- [ ] A destination is configured, and `/health → audit_backup.configured` is true
- [ ] `audit_backup.off_pod` / `filesystem.same_volume` say the destination is really off the pod
- [ ] One manual backup pass has succeeded and objects are visible in the bucket/dir
- [ ] `--require-destination` is set on the scheduled job
- [ ] `VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS` runs before rotation can delete a file
- [ ] Remote retention is a bucket lifecycle rule, or `--prune` is scheduled
- [ ] `audit.write_failures` is 0 and `disk.below_floor` is false
- [ ] `VA_LSE_AUDIT_ERROR_MESSAGES=0` if the destination is third-party storage
- [ ] Restore rehearsed: download the most recent object and confirm it parses as JSON-lines

---

## 17. LLM endpoint failover (optional)

**Single-endpoint deployments are fully supported and are the default.** If
`OPENAI_BASE_URL_FALLBACK` is unset, the app runs on exactly one endpoint, with no
second probe and no change in behaviour. When the endpoint is down, runs fail with a
fast, actionable error and users retry — see *Manual failover* below for the steps an
operator takes.

Configuring a second endpoint buys continuity through a provider outage without giving
up the primary: calls are served by the backup only while the primary is genuinely
broken, and return to the primary automatically.

### What happens, and when

| Time since the primary started failing | What the app does |
|---|---|
| 0 – ~60s (below the breaker threshold ×3) | Retries within each call. Nothing visible. This is the window that `LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=0` does **not** remove: the breaker must open before anything counts as an outage, so the failures that open it still fail. |
| Breaker opens | Calls fail fast (`CircuitBreakerOpenError`) in under 2s without touching the network, so users are not made to wait behind a dead endpoint. |
| Up to `LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS` (default 300s) | **Still only the primary.** This grace period is the point: a blip, a provider blip, or a rate-limit storm should not move the business onto another provider. The error message says when failover will engage. |
| Past the grace period | Calls are served by `OPENAI_BASE_URL_FALLBACK`. Users keep working. The run is **stamped** (`llm_endpoints` in the audit record and the run log), and `va_lse_llm_failover_active` becomes `1`. |
| While failed over | The primary's own recovery window (60s) keeps elapsing, and the next real call is tried there first. If it succeeds, traffic returns to the primary; if it fails, **that same call is served by the fallback**, so a recovery probe never turns into a user-visible error. |
| Primary answers again | Its breaker closes and the unhealthy clock clears. All traffic is on the primary from the next call. |

Two properties of this design are worth keeping if it is ever reimplemented:

- **The failover trigger is not the breaker's recovery timer.** That timer is reset by
every failed probe, so with traffic flowing it never ages past one recovery timeout —
a rule written as "the breaker has been OPEN for 5 minutes" would fire only on an idle
system, i.e. never during the outage it exists for. `unhealthy_for_seconds()` is a
separate clock, cleared only by a genuine recovery.
- **A probe cannot break a user's run.** The primary is re-tested with real traffic and
a failure falls through to the fallback inside the same call.

### Arming it

```bash
# Minimum: a second endpoint under the same account (same key, same model names).
OPENAI_BASE_URL_FALLBACK=https://second-gateway.example.com/v1

# A different provider needs its own key and its own model names.
OPENAI_BASE_URL_FALLBACK=https://api.openai.com/v1
OPENAI_API_KEY_FALLBACK=sk-proj-...
LLM_MODEL_MAIN_FALLBACK=gpt-4-turbo
LLM_MODEL_FAST_FALLBACK=gpt-4o-mini

# How long the primary must fail before failover engages (a grace period, NOT an
# HTTP timeout — that is VA_LSE_LLM_CALL_TIMEOUT_SECONDS). 0 = no grace period:
# failover engages as soon as the primary's breaker opens, i.e. after
# VA_LSE_CB_FAILURE_THRESHOLD consecutive failures. The failures that trip the
# breaker still fail; 0 does not prevent them.
LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=300
```

Unset fallback values inherit the primary's, so a second gateway under one account
needs a single variable.

The fallback is configured through the **environment or secret store only**, not the
sidebar. That is deliberate rather than an omission: in Pattern C a worker 
builds its own client from the environment (a value typed into a browser session
cannot reach it), so a fallback that existed only for the web tier would be worse
than none — it would appear armed and silently not apply to queued runs. The
primary's own sidebar fields keep working as before.

Verify on a running instance:

```bash
curl -s localhost:8001/health | jq .llm_failover
# { "configured": true, "active": false, "after_seconds": 300, "primary_unhealthy_seconds": 0 }
```

`configured: true` with `active: false` is the normal state — armed, unused. The
fallback URL is validated at client construction, so a typo or a missing model name is
reported at startup rather than discovered mid-outage.

Pointing both endpoints at the **same URL** is allowed but logs a warning: it cannot be
failover, because the same endpoint fails the same way.

### Readiness and alerting

Readiness asks "can this instance serve a run?", so while a healthy fallback is serving,
`/ready` stays **200** and the pod stays in the load balancer. The primary outage is
reported instead of hidden:

- `/health → llm_failover` (state above)
- `va_lse_llm_failover_active == 1`, `va_lse_llm_primary_unhealthy_seconds`
- `VaElseLlmRunningOnFallback` and `VaElseLlmPrimaryUnhealthy` in `deploy/monitoring/alerts.yml`
- `va_lse_llm_endpoint_calls_total` / `..._duration_ms` split by `endpoint`, so you can
  see the backup's volume and latency while it carries traffic

This separation is deliberate: failing readiness during a *handled* failover would drain
the pod and page on-call for a problem that is already being compensated for. The
trade-off is that a fallback outage is the only thing that turns `llm_failover` into a
failing readiness — which is the correct signal, since at that point nothing can serve.

### Manual failover (single-endpoint deployments)

Without a configured fallback, an extended primary outage is a manual operation. Steps,
in order:

1. **Confirm it is the endpoint, not your credentials or the model.**
   `curl -s localhost:8001/health | jq .llm_failover.primary_unhealthy_seconds` and
   `jq .llm_circuit_breaker_open`; then check the provider's status page. A
   `client`-category error spike means a bad key or model name, which failover cannot
   fix either.
2. **Decide the replacement endpoint** and its two model names, and confirm the key is
   valid with `curl -H "Authorization: Bearer $KEY" $BASE_URL/models`.
3. **Set the values** — `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL_MAIN`,
   `LLM_MODEL_FAST` — in the deployment's environment or secret store. In
   **Pattern C the worker reads the environment only**: a key typed into the sidebar
   cannot reach it, and the submit path refuses a job it knows the worker cannot run.
4. **Restart the affected processes.** The client is built from settings, so:
   - web pod / container: a **sidebar → Apply settings** change takes effect on the next
     script run, with no restart (the process-global breaker however stays OPEN for up
     to its recovery window, or until a call succeeds);
   - workers, and any environment change: re-create the pod(s) so the new values are read.
5. **Verify**: `/ready` returns 200, then run a single small Evaluate and confirm
   `llm_endpoints: ["primary"]` in the new audit record.
6. **When the original provider recovers**, reverse steps 3–4. If the outage is over, do
   not leave the deployment pointed at a provider you did not choose deliberately.

### Rehearsing the outage

A failover path that has never moved traffic is a hypothesis, and the moment you
find out it does not work is the moment it was supposed to save you. Rehearsing
costs one restart and no LLM tokens, because the trigger is *making the primary
unreachable*, not breaking a credential — the circuit breaker reacts to transport
failure, and an unusable key would fail the same way but take the fallback's
health check down with it if they share one.

`scripts/rehearse_failover.py` reads the probe port only (`/health` and
`/metrics`), so it needs no credentials, sends no LLM traffic, and cannot disturb
a run in progress:

```bash
python scripts/rehearse_failover.py                       # one snapshot, human readable
python scripts/rehearse_failover.py --expect-idle         # step 1 and 5 below
python scripts/rehearse_failover.py --expect-active       # step 3
python scripts/rehearse_failover.py --watch --interval 5  # follow the stages live
python scripts/rehearse_failover.py --json | jq .active    # scripted check
```

Exit codes: `0` consistent and matching `--expect-*`, `1` unreadable or
self-contradictory, `2` readable but not the stage asserted. `--json` prints
**only** the JSON document, so it stays pipeable; the `--expect-*` verdict is still
carried in the exit code.

The drill:

1. `--expect-idle` — the backup is armed and unused.
2. Cause the outage: point `OPENAI_BASE_URL` at an unroutable host, or block egress
   to the provider from the web **and** worker pods. Watch with `--watch`. Expect the
   primary's breaker to go `OPEN`, `primary_unhealthy_seconds` to start growing, and
   `active` to flip to `1` after `LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS`.
   Before that threshold a call still **fails fast** — that is the documented
   contract, and the sidebar's failover panel shows the countdown.
3. `--expect-active` — traffic is being served by the backup.
4. Restore the primary and watch again. Expect `active` to return to `0` and the
   unhealthy clock to reset **without a restart**: the primary is re-tested with
   real traffic, so recovery is detected by the next call.
5. Undo step 2 and confirm `--expect-idle` passes again.

Submit one small Evaluate while failed over and check the run's audit record — it
must carry `llm_endpoints: ["primary", "fallback"]`. That stamp is the only
permanent evidence of which provider produced a document, so a drill that stops
at `active` has not verified the part that matters.

To do this against a real cluster instead of a local stack, point the tool at the
probe port from inside the cluster (`kubectl run --rm -it` on the app image, or a
port-forward) and run the same five steps:

```bash
kubectl port-forward -n va-lse svc/va-lse-web 8001:8001 &
python scripts/rehearse_failover.py --expect-idle
```

### Failover checklist

- [ ] You are running single-endpoint **on purpose**, or `OPENAI_BASE_URL_FALLBACK` is set
- [ ] The fallback key and both model names are set if it is a different provider
- [ ] `/health → llm_failover.configured` is `true` and `active` is `false`
- [ ] `LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS` matches your tolerance for a primary blip
- [ ] Readiness returns 200 with a healthy primary (no false 503s introduced)
- [ ] `VaElseLlmRunningOnFallback` is routed to someone who can act on it
- [ ] You have checked what the backup provider costs, and its rate limit fits `VA_LSE_MAX_CONCURRENT_LLM_CALLS`
- [ ] You know the run output may differ slightly while failed over, and that the audit
      record's `llm_endpoints` is where that is recorded
- [ ] `python scripts/rehearse_failover.py --expect-idle` exits 0 against the deployment
- [ ] You have run the outage drill at least once, and seen `--expect-active` pass
      mid-outage and `--expect-idle` pass again after recovery, with no restart
- [ ] The manual steps above are in your runbook if you deploy single-endpoint

---


## Appendix: Checklist for production deployment

- [ ] `.env` is NOT committed to git (see `SECURITY.md`)
- [ ] Secrets are stored in K8s Secret / Docker secret / cloud secret manager
- [ ] On a host with no `.env` (Streamlit Cloud), `OPENAI_API_KEY` **and** `OPENAI_BASE_URL` are
      set together in **Settings → Secrets** and the app was rebooted afterwards ([§14](#14-pattern-d--streamlit-community-cloud))
- [ ] Sidebar **Test connection** reports a real call answered against the deployed base URL
- [ ] Sidebar field(s) sourced from secrets are captioned (`🔐 From Streamlit secrets: …`)
- [ ] Any API key pasted into a chat, issue, or commit has been rotated (`SECURITY.md`)
- [ ] `OPENAI_API_KEY` is set and valid
- [ ] `VA_LSE_LOG_DIR` and `VA_LSE_AUDIT_LOG_DIR` point to a persistent volume
- [ ] Health probes are configured in the Deployment/Pod spec
- [ ] `terminationGracePeriodSeconds` ≥ `VA_LSE_SHUTDOWN_GRACE_SECONDS + 15`
- [ ] TLS termination is handled by a reverse proxy (not Streamlit)
- [ ] Security headers (CSP, HSTS, X-Frame-Options) are set by the proxy
- [ ] `.streamlit/config.toml` is present in the container (XSRF, toolbar, maxUploadSize)
- [ ] LLM endpoint can handle `N_pods × VA_LSE_MAX_CONCURRENT_LLM_CALLS` concurrent requests
- [ ] Audit logs (`logs/audit.log`) are retained per your compliance policy
- [ ] Circuit breaker + concurrency limiter env vars are tuned for your user count
- [ ] Shared cache (`VA_LSE_SHARED_CACHE_URL/TOKEN`) is configured for multi-instance deployments
- [ ] Cache hit rate is monitored via `GET /health` → `cache.hit_rate`
- [ ] Tracing configured on both tiers, or deliberately left off ([§15](#15-distributed-tracing-opentelemetry))
- [ ] `python scripts/restore_audit_logs.py --verify` reports `VERDICT: ok` (a backup that has never been read back is a hypothesis)
- [ ] The audit backup destination is off-pod (`/health` → `audit_backup.off_pod` is `true`)
- [ ] `GET /metrics` is scraped, with an alert on `va_lse_audit_backup_state > 1`
- [ ] `VA_LSE_AUDIT_ERROR_MESSAGES=0` if audit logs go to a third-party destination
- [ ] The LLM failover drill has been run to completion ([§17](#17-llm-endpoint-failover-optional)) — an
      untested failover path fails on the day it is needed, not before
