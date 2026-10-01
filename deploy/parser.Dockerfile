# Reviewed parser-only image: no web app, prompts, secrets, logs or other cases.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY app/__init__.py app/config.py app/documents.py app/parser_protocol.py app/parser_worker.py ./app/
ARG VA_LSE_BUILD_SHA
ENV VA_LSE_BUILD_SHA=${VA_LSE_BUILD_SHA}
USER 65534:65534
ENTRYPOINT ["python", "-m", "app.parser_worker"]
