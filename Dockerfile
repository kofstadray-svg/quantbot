# ── Stage 1: dependencies ─────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps needed by some Python packages (ta-lib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code (excludes .venv, logs, .git via .dockerignore)
COPY . .

# Logs volume — mount to persist across restarts
VOLUME ["/app/logs"]

# Health port is service-specific; set via CMD / docker-compose
EXPOSE 9100 9101 9102

# Default: bot scheduler. Override in docker-compose per service.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=dev

CMD ["python", "main.py"]
