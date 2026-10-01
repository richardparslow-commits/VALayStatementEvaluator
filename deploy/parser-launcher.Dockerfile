# Trusted launcher. Docker authority is restricted to this service, never the web/parser.
FROM docker:28-cli AS docker-cli
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY app/__init__.py app/parser_protocol.py app/parser_service.py ./app/
RUN mkdir -p /run/parser && chown 65534:65534 /run/parser && chmod 700 /run/parser
USER 65534:65534
ENTRYPOINT ["python", "-m", "app.parser_service"]
