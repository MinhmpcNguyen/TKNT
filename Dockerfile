# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY app-config.docker.yaml /app/app-config.yaml
COPY requirements.txt /app/requirements.txt
COPY api /app/api
COPY agent /app/agent
COPY agent_schema /app/agent_schema
COPY assets_gen /app/assets_gen
COPY clients /app/clients
COPY cluster_placer /app/cluster_placer
COPY config /app/config
COPY constraint /app/constraint
COPY db /app/db
COPY layout /app/layout
COPY object_program /app/object_program
COPY pipeline /app/pipeline
COPY prompt /app/prompt
COPY query /app/query
COPY services /app/services
COPY stylist /app/stylist
COPY synthetic_data /app/synthetic_data
COPY tier_count /app/tier_count

RUN python -c "import api.main"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html frontend/tsconfig.json frontend/tsconfig.node.json frontend/vite.config.ts ./
COPY frontend/src ./src
COPY frontend/public/assets/three ./public/assets/three

ENV VITE_API_BASE=/api
ENV TKNT_VITE_PUBLIC_DIR=false
RUN npm run build

FROM nginx:1.27-alpine AS web

COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

EXPOSE 80
