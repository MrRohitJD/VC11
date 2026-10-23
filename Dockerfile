FROM node:22-alpine AS frontend
WORKDIR /build
COPY gmail-ai-dashboard/frontend/package*.json ./
RUN npm ci
COPY gmail-ai-dashboard/frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY gmail-ai-dashboard/backend/requirements-heroku.txt ./requirements-heroku.txt
RUN pip install --no-cache-dir -r requirements-heroku.txt
COPY gmail-ai-dashboard/backend/ ./
COPY --from=frontend /build/dist /app/frontend/dist
CMD ["sh", "-c", "export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false; exec uvicorn app.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
