# ─────────────────────────────────────────────────────────────────────────────
# SmartDoc Chatbot — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t smartdoc-chatbot .
# Run UI: docker run -p 8080:8080 --env-file .env smartdoc-chatbot
# Run API:docker run -p 8000:8000 --env-file .env smartdoc-chatbot uvicorn api:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies (gcc required for some C-extension wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Expose ports ──────────────────────────────────────────────────────────────
#   8080 → Chainlit UI
#   8000 → FastAPI backend
EXPOSE 8080 8000

# ── Default command: run Chainlit UI ─────────────────────────────────────────
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
