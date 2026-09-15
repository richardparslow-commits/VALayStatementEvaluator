# Deployment Guide — VA Lay Statement Evaluator

This document covers running the app in production at scale, including
multi-instance deployment behind a load balancer, session persistence,
and graceful failover. For single-user local setup, see `README.md → Setup`.

## Table of contents

1. [Deployment patterns at a glance](#1-deployment-patterns-at-a-glance)
2. [Pattern A — Docker Compose + nginx (session affinity)](#2-pattern-a--docker-compose--nginx-session-affinity)
3. [Pattern B — Kubernetes with session affinity](#3-pattern-b--kubernetes-with-session-affinity)
4. [Pattern C — Kubernetes with Redis session store](#4-pattern-c--kubernetes-with-redis-session-store)
5. [Session persistence tradeoffs](#5-session-persistence-tradeoffs)
6. [Dockerfile](#6-dockerfile)
7. [Health probe wiring](#7-health-probe-wiring)
8. [Graceful shutdown at scale](#8-graceful-shutdown-at-scale)
9. [Environment variables reference](#9-environment-variables-reference)
10. [Scaling guidance](#10-scaling-guidance)
11. [Rate limiting (reverse proxy)](#11-rate-limiting-reverse-proxy)
12. [TLS and reverse proxy](#12-tls-and-reverse-proxy)
13. [Distributed cache for VA reference data](#13-distributed-cache-for-va-reference-data)

---

## 1. Deployment patterns at a glance

| Pattern | Instances | Session persistence | Failover | Complexity |
|---|---|---|---|---|
| **A. Docker Compose + nginx** | 3 (configurable) | Cookie-based affinity | Affinity preserves session; new instance loses state | Low — ideal for small teams |
| **B. K8s + session affinity** | ≥ 2 via Deployment | Client-IP or cookie affinity | Same-node sessions survive pod restart; cross-node sessions lost | Medium |
| **C. K8s + Redis session store** | ≥ 2 via Deployment | Redis-backed `st.session_state` | Full: any pod serves any session; pod kill is invisible to user | High — best for 100-user scale |

**Recommendation for 100 concurrent users:** Pattern C (Kubernetes + Redis). Session
affinity (Patterns A/B) creates hot spots — a user who uploads a large record set
keeps hitting the same pod, preventing the load balancer from spreading work. Redis
eliminates this constraint. Patterns A/B are suitable for <20 users or development/staging.

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

## 4. Pattern C — Kubernetes with Redis session store

This is the recommended pattern for 100 concurrent users. It replaces
in-memory session state with Redis, so any pod can serve any session.

### How it works

Streamlit stores per-session data in `st.session_state` (a Python dict).
To externalize this:

1. **`streamlit-ext-session-state`** (or a custom wrapper around
   `streamlit-javascript` + Redis) serializes session state to Redis on
   every rerun and deserializes it at the start.
2. **Alternative (simpler, recommended):** Keep the app stateless per-request
   and reconstruct it on each Streamlit rerun from persistent storage. The
   app already does this for most state:
   - Uploaded documents are re-extracted from the uploaded files (cached
     per `st.session_state` but re-extractable).
   - Evaluation/Draft results are stored in `st.session_state` and lost on
     pod migration — but the user can re-run.
   - The `request_id` and audit trail are written to `logs/audit.log` (not
     session state), so they survive pod restarts.

3. **Pragmatic approach:** Use a **Redis-backed session store** that persists
   `st.session_state` across pod restarts, so mid-run pod kills do not lose
   the user's uploaded records and in-progress results.

### Redis session store pattern

```yaml
# k8s-redis.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: va-lse-redis
spec:
  serviceName: va-lse-redis
  replicas: 1
  selector:
    matchLabels:
      app: va-lse-redis
  template:
    metadata:
      labels:
        app: va-lse-redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          command: ["redis-server", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]
          volumeMounts:
            - name: redis-data
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: redis-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: va-lse-redis
spec:
  selector:
    app: va-lse-redis
  ports:
    - port: 6379
```

The Streamlit app would need a session-state hook to persist to Redis:

```python
# In app/main.py or a new app/session_store.py:
import json
import os

REDIS_URL = os.getenv("VA_LSE_REDIS_URL", "")

def _sync_session_to_redis() -> None:
    """Persist st.session_state to Redis after every rerun (best-effort)."""
    if not REDIS_URL:
        return
    try:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        session_id = st.session_state.get("_va_lse_session_id", "")
        if not session_id:
            import uuid
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            st.session_state["_va_lse_session_id"] = session_id
        # Serialize session state (exclude non-serializable items)
        serializable = {}
        for k, v in st.session_state.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                pass  # Skip non-serializable items (e.g. UploadedFile objects)
        r.setex(f"va_lse:session:{session_id}", 7200, json.dumps(serializable))  # 2h TTL
    except Exception:
        pass  # Redis failure must never break the app

def _load_session_from_redis() -> None:
    """Load session state from Redis on app start (best-effort)."""
    if not REDIS_URL:
        return
    try:
        import redis
        import uuid
        r = redis.from_url(REDIS_URL, decode_responses=True)
        session_id = st.session_state.get("_va_lse_session_id", "")
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            st.session_state["_va_lse_session_id"] = session_id
        data = r.get(f"va_lse:session:{session_id}")
        if data:
            for k, v in json.loads(data).items():
                if k not in st.session_state:
                    st.session_state[k] = v
    except Exception:
        pass
```

### Integration with the app

Add to `app/config.py`:

```python
# Redis session store (optional). Set VA_LSE_REDIS_URL to enable
# Redis-backed session persistence for multi-instance deployments.
# Without it, sessions are in-memory per pod (Pattern A/B).
REDIS_URL = os.getenv("VA_LSE_REDIS_URL", "")
```

Add to `.env.example`:

```bash
# Redis-backed session store for multi-instance K8s deployments (Pattern C).
# Leave empty for single-instance or affinity-based deployments (Patterns A/B).
# VA_LSE_REDIS_URL=redis://va-lse-redis:6379/0
```

### Deployment

```bash
# Create secret
kubectl create secret generic va-lse-env --from-env-file=.env

# Add VA_LSE_REDIS_URL to the secret
kubectl patch secret va-lse-env -p \
  '{"data":{"VA_LSE_REDIS_URL":"cmVkaXM6Ly92YS1sc2UtcmVkaXM6NjM3OS8w"}}'  # base64-encoded

# Apply all manifests
kubectl apply -f deploy/k8s/
```

---

## 5. Session persistence tradeoffs

| Factor | Pattern A/B (affinity) | Pattern C (Redis) |
|---|---|---|
| **Setup complexity** | Low — just nginx cookie or k8s affinity annotation | Medium — Redis StatefulSet + session hook code |
| **Failover behavior** | User loses session if their pod dies; must re-upload records and re-run | Session survives pod restart; user sees a brief pause |
| **Load distribution** | Hot spots — large record sets keep one pod busy while others idle | Even — any pod can pick up any session |
| **Memory per pod** | Higher (session state accumulates) | Lower (state offloaded to Redis) |
| **Operational cost** | Zero additional infra | Redis pod + 1 GB PVC + monitoring |
| **When to use** | < 20 users, dev/staging, quick demos | 20–100+ concurrent users, production |

**Key insight:** The app's heaviest state — uploaded medical records and
`MedicalDigest` — is already re-extractable from the source files. The only
truly volatile state is in-progress pipeline results and the `request_id`.
For many deployments, **Pattern B with session affinity is sufficient** because:

1. Pod restarts are rare in healthy clusters (kubectl drain + rolling updates
   are the main causes).
2. When they do occur, the user sees a Streamlit error page, refreshes, and
   re-uploads their files (they already have the record bundle locally).
3. The audit trail (`logs/audit.log`) survives because it writes to a
   persistent volume, not session state.

Pattern C is worth the complexity when you need:
- Zero-interruption failover (e.g., rolling upgrades during business hours)
- Pod autoscaling (HPA adds/removes pods based on CPU; sessions must be
  transferable)
- Compliance requirements that mandate session audit continuity

---

## 6. Dockerfile

The production Dockerfile is optimized for the smallest possible image and
fastest startup. It installs from the hash-pinned lockfile and runs as a
non-root user.

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
| `OPENAI_BASE_URL` | LLM endpoint | QwenCloud Token Plan | Verify against your provider |
| `LLM_MODEL_MAIN` | Analysis model | `qwen3.7-max` | Match your plan |
| `LLM_MODEL_FAST` | Bulk digest model | `qwen3.7-flash` | Match your plan |
| `VA_LSE_RECORDS_CONCURRENCY` | Parallel digest workers | `2` | Raise for higher-tier endpoints |
| `VA_LSE_MAX_CONCURRENT_LLM_CALLS` | Global LLM concurrency cap | `20` | Raise if running 100 users across N pods |
| `VA_LSE_HEALTH_PORT` | Health sidecar port | `8001` | Keep default; mount in Service |
| `VA_LSE_SHUTDOWN_GRACE_SECONDS` | Drain timeout | `30` | 30–60 for large record sets |
| `VA_LSE_LLM_CALL_TIMEOUT_SECONDS` | Per-call timeout | `300` | 300–600 depending on endpoint speed |
| `VA_LSE_LOG_DIR` | Diagnostic log directory | (stdout only) | Set to `/app/logs` for persistent logs |
| `VA_LSE_AUDIT_LOG_DIR` | Audit log directory | `logs` | Set to `/app/logs` for persistent logs |
| `VA_LSE_LOG_JSON` | JSON log format | `0` | `1` for ELK/CloudWatch/Datadog |
| `VA_LSE_REDIS_URL` | Redis session store | (empty) | `redis://va-lse-redis:6379/0` for Pattern C |

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
| `st.session_state` | In-memory | No (Pattern A/B) / Yes (Pattern C with Redis) | No / Yes (Redis) |
| `request_id` | `st.session_state` + `ContextVar` | No | No |
| Audit logs | `logs/audit.log` (PV) | Yes | Yes (if shared PVC) |
| Diagnostic logs | `logs/app.log` (PV) | Yes | Yes (if shared PVC) |
| `usage_history.json` | Filesystem | No | No (per-pod) |
| Circuit breaker | In-memory | No | No (per-pod is correct) |
| Health cache | In-memory | No | No (per-pod is correct) |

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

Beyond IP-level rate limiting, enforce per-user concurrency limits in
the Streamlit app itself. The circuit breaker and concurrency limiter
(`app/circuit_breaker.py`) already cap global LLM calls per pod, but
per-session limits prevent one user from monopolizing a pod:

```python
# In app/main.py (already implemented via session state):
# - One concurrent Evaluate run per session (st.session_state['eval_running'])
# - One concurrent Draft run per session (st.session_state['draft_running'])
# - Shutdown gate rejects new runs during drain
```

These are enforced by the Streamlit rerun model — clicking "Run" while a
current run is in progress is a no-op (the button is disabled). For
additional server-side enforcement:

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

---

## Appendix: Checklist for production deployment

- [ ] `.env` is NOT committed to git (see `SECURITY.md`)
- [ ] Secrets are stored in K8s Secret / Docker secret / cloud secret manager
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
