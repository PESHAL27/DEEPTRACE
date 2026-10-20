# ==============================================================================
# DEEPTRACE — Production Single-Node Container Deployment
# Multi-stage build packaging React/Vite frontend and FastAPI/PyTorch CPU backend.
# ==============================================================================

# Stage 1: Build Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json tsconfig*.json vite.config.ts index.html ./
RUN npm ci --prefer-offline --no-audit
COPY public/ ./public/
COPY src/ ./src/
ENV VITE_API_BASE_URL=/api
ENV VITE_MEDIA_BASE_URL=
RUN npm run build

# Stage 2: Production Python Backend Runtime
FROM python:3.12-slim AS runner

# Install essential system libraries for OpenCV and multimedia processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python runtime dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend application source
COPY backend/app/ ./backend/app/
COPY backend/evaluation/ ./backend/evaluation/

# Copy model checkpoints and DeepfakeBench architecture definitions
COPY deepfake_model/ ./deepfake_model/

# Copy built frontend distribution into static mount target
COPY --from=frontend-builder /app/dist ./dist

# Create runtime directories for persistence
RUN mkdir -p /app/backend/uploads/thumbnails \
             /app/backend/uploads/heatmaps \
             /app/backend/uploads/forensics/frequency \
             /app/backend/uploads/reports \
             /app/backend/results

# Configure environment defaults
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    UPLOADS_DIR=/app/backend/uploads \
    RESULTS_DIR=/app/backend/results

# Healthcheck targeting the readiness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/api/status || exit 1

EXPOSE 8000

WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
