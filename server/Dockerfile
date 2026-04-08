FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

CMD ["sh", "-c", "uvicorn hr.server.app:app --host $HOST --port $PORT --workers $WORKERS"]
