# CHROME — root-level Dockerfile (required by Hugging Face Spaces)
# Builds and serves the FastAPI environment on port 8000.

FROM python:3.10-slim as builder

# git is required for the git+ dependency in pyproject.toml
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire repo
COPY . .

# Install with all dependencies (core + inference extras)
RUN pip install --no-cache-dir -e ".[inference]"

# HF Spaces runs as non-root; ensure /app is writable
RUN chmod -R 755 /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server via the installed package path
CMD ["python", "-m", "uvicorn", "hr.server.app:app", "--host", "0.0.0.0", "--port", "8000"]