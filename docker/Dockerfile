# syntax=docker/dockerfile:1.6
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Without this the app falls back to development mode and boots with the public
# placeholder secret key. Production mode makes a missing FLASK_SECRET_KEY fatal.
ENV FLASK_ENV=production

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY wsgi.py ./
# Both are named by pyproject.toml (readme, license); without them the installed
# package metadata in the image is missing its long description and licence.
COPY README.md LICENSE ./

# --no-deps: requirements.txt above already resolved every dependency at a pinned
# version. A plain `pip install .` would re-resolve them and could drift off the pins.
RUN pip install --no-deps .

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:8080/healthz > /dev/null || exit 1

# One worker only: the price cache, its RLock, and the memory:// rate limiter are
# all per-process, so a second worker halves cache hits and doubles the real limit.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "wsgi:application"]
