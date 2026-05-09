# syntax=docker/dockerfile:1.6
#
# Production image for the Financial Mathematical Models web app.
# Serves the Flask app via gunicorn behind port 8000.

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages required by numpy / scipy wheels and yfinance's HTTP stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so they cache independently of source changes.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application source.
COPY app ./app
COPY portfolio ./portfolio
COPY wsgi.py ./

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:8000/healthz > /dev/null || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "wsgi:application"]
