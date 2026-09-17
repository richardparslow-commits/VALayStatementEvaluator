# Kubernetes Manifests — VA Lay Statement Evaluator

This directory contains raw Kubernetes manifests for deploying the app.
For the full deployment guide, the placement tradeoff analysis, and the Pattern C
worker-pool setup, see `DEPLOYMENT.md`.

`k8s-redis.yaml` and `k8s-worker.yaml` are the **Pattern C** tier: a Redis-backed
job queue plus worker pods that run the digest pipeline. They are only needed when
you enable `VA_LSE_JOB_QUEUE=1`; without them the app runs every run in-process
inside the Streamlit pod, which is fine for small deployments.

## Quick start

```bash
# 1. Create secrets from .env
kubectl create secret generic va-lse-env --from-env-file=../../.env

# 2. Apply manifests (order matters: namespace → configmap → deployment → service → ingress)
kubectl apply -f k8s-namespace.yaml
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-ingress.yaml        # optional, needs ingress controller

# 3. Watch pods
kubectl get pods -n va-lse -l app=va-lse -w

# 4. Check health
kubectl port-forward -n va-lse svc/va-lse 8001:8001
curl http://localhost:8001/health
curl http://localhost:8001/ready

# 5. Access the app
kubectl port-forward -n va-lse svc/va-lse 8501:80
# Open http://localhost:8501
```

## Files

| File | Description |
|---|---|
| `k8s-namespace.yaml` | Dedicated namespace for isolation |
| `k8s-deployment.yaml` | 3-replica Deployment with health probes + graceful shutdown |
| `k8s-service.yaml` | ClusterIP Service for internal access |
| `k8s-ingress.yaml` | Ingress with cookie-based session affinity (requires nginx-ingress) |
| `k8s-hpa.yaml` | Horizontal Pod Autoscaler (CPU-based, 3–20 replicas) |
| `k8s-redis.yaml` | Redis StatefulSet backing the Pattern C job queue |
| `k8s-worker.yaml` | Worker Deployment that executes queued runs (Pattern C) |
| `k8s-logs.yaml` | ReadWriteMany PVC for the audit + run logs (mount it before applying the rest) |
| `k8s-audit-backup.yaml` | CronJob that ships the audit log off-pod and enforces retention |

> **Apply `k8s-logs.yaml` first.** Both Deployments mount `va-lse-logs` at `/app/logs`,
> so without it they stay in `ContainerCreating` (a Pending PVC). It replaces an
> `emptyDir` that reclaimed the entire audit stream on every restart — and, in
> Pattern C, kept the web tier's audit log separate from the workers'. Nothing can back
> up a volume that disappears with its pod, so this is the prerequisite for
> [`k8s-audit-backup.yaml`](#audit-log-backup), not an optional extra.

## Pattern C (worker pool)

```bash
# 1. Queue backend + workers, BEFORE enabling the queue on the web tier
kubectl apply -f k8s-redis.yaml -n va-lse
kubectl apply -f k8s-worker.yaml -n va-lse
kubectl rollout status deployment/va-lse-worker -n va-lse

# 2. Confirm a worker is serving and the queue is reachable
kubectl port-forward -n va-lse deploy/va-lse-worker 8002:8002
curl -s localhost:8002/health | grep -A5 job_queue

# 3. Turn the queue on for the web tier
kubectl set env deployment/va-lse -n va-lse VA_LSE_JOB_QUEUE=1

# 4. Watch a run land: the web pod should stay near-idle during a large digest
kubectl top pods -n va-lse
```

Rollback is `kubectl set env deployment/va-lse -n va-lse VA_LSE_JOB_QUEUE=0`.
Jobs already queued still finish; new runs execute in-process again.

Workers need `OPENAI_API_KEY` in `va-lse-env` **or a mounted secret** — a key
typed into the sidebar cannot reach a worker process, and the app will refuse to
queue a run rather than fail it on the worker.

## Audit log backup

```bash
# 0. The shared log volume must exist first (see k8s-logs.yaml)
kubectl apply -f k8s-logs.yaml -n va-lse
kubectl get pvc va-lse-logs -n va-lse          # Bound, not Pending

# 1. A destination, then the job
kubectl create secret generic va-lse-backup-creds -n va-lse \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...
kubectl apply -f k8s-audit-backup.yaml

# 2. Prove it works now, not at 3am (a scheduled job that has never succeeded
#    looks the same as one that was never scheduled — /health calls that "never_ran")
kubectl create job --from=cronjob/va-lse-audit-backup audit-backup-manual -n va-lse
kubectl logs -n va-lse job/audit-backup-manual
curl -s localhost:8001/health | python -m json.tool | grep -A8 audit_backup
```

The CronJob runs every 6 hours (`VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS`), passes
`--prune` for remote retention and `--require-destination` so a deployed-but-
unconfigured job **fails visibly** instead of exiting 0 forever. Prefer workload
identity over the static keys above. Full procedure, retention table, and the
`error_message` egress caveat: [DEPLOYMENT.md §16](../../DEPLOYMENT.md#16-audit-log-retention-and-backup).

## Verifying the backup is readable

A green CronJob means the uploads succeeded, not that the objects can be read back.
Run this from a pod (or anywhere with the credentials) monthly — it is the check that
catches an upload writing objects nothing can restore:

```bash
kubectl run audit-verify -n va-lse --rm -it --restart=Never \
  --image=<your-web-image> -- python scripts/restore_audit_logs.py --verify
```

Exit `0` means every object hashed correctly and the byte ranges tile the stream with
no holes; `2` names the corrupt objects or the exact missing offsets. It never writes
to the destination.

## Metrics for alerting

Both tiers expose Prometheus text format on their existing health port:

```bash
curl -s localhost:8001/metrics | grep va_lse_audit_backup
curl -s localhost:8002/metrics | grep va_lse_job_queue
```

Scrape `:8001/metrics` on the web pods and (for Pattern C) `:8002/metrics` on the
workers; annotate the Deployments with `prometheus.io/scrape`, `prometheus.io/port`,
and `prometheus.io/path` if you run the Prometheus operator. Alert rules and the full
metric list: [DEPLOYMENT.md §16](../../DEPLOYMENT.md#metrics-for-alerting-get-metrics).
The scrape performs no network I/O, so a slow Redis tier degrades the queue rather
than the metrics (and liveness) endpoint.

## Scaling

```bash
# Manual scale — web tier (sticky UI pods)
kubectl scale deployment va-lse --replicas=10 -n va-lse

# Or let HPA manage it (apply k8s-hpa.yaml)
kubectl apply -f k8s-hpa.yaml

# Worker tier scales independently of the web tier. Throughput is bounded by the
# LLM endpoint, not pod count, so watch rate limits before scaling out.
kubectl scale deployment va-lse-worker --replicas=4 -n va-lse
```

## Secrets

```bash
# Create from .env
kubectl create secret generic va-lse-env --from-env-file=../../.env -n va-lse

# Update after .env changes
kubectl delete secret va-lse-env -n va-lse
kubectl create secret generic va-lse-env --from-env-file=../../.env -n va-lse
kubectl rollout restart deployment/va-lse -n va-lse
```
