# Kubernetes Manifests — VA Lay Statement Evaluator

This directory contains raw Kubernetes manifests for deploying the app.
For the full deployment guide, tradeoff analysis, and Redis session store
option, see `DEPLOYMENT.md`.

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
| `k8s-redis.yaml` | Redis StatefulSet for session persistence (Pattern C) |

## Scaling

```bash
# Manual scale
kubectl scale deployment va-lse --replicas=10 -n va-lse

# Or let HPA manage it (apply k8s-hpa.yaml)
kubectl apply -f k8s-hpa.yaml
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
