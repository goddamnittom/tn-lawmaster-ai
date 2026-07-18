# ─────────────────────────────────────────────
# TN-LawMaster — Multi-stage Production Docker Image
# ─────────────────────────────────────────────
# Stage 1: Build (install dependencies)
FROM python:3.12-slim AS builder

WORKDIR /app

# Build tools for packages that compile C extensions (e.g. chromadb, pymupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime (minimal image, no build tools)
FROM python:3.12-slim

WORKDIR /app

# Non-root user for security
RUN groupadd -r tnapp && useradd -r -g tnapp -d /app tnapp

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=tnapp:tnapp . .

# Ensure data and log directories exist with correct ownership
RUN mkdir -p data logs chroma_db && chown -R tnapp:tnapp data logs chroma_db

USER tnapp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main_fastapi:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
