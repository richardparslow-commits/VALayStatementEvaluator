# Trusted launcher; private mTLS authority belongs only to a dedicated parser VM.
FROM docker:28-cli AS docker-cli
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY app/__init__.py app/ingestion_policy.py app/parser_protocol.py app/parser_engine.py app/parser_service.py ./app/
RUN mkdir -p /run/parser && chown 65534:65534 /run/parser && chmod 700 /run/parser
USER 65534:65534
ENTRYPOINT ["python", "-m", "app.parser_service"]
