# DEEPTRACE — Production Deployment & Operations Guide

## Executive Summary
This document provides the authoritative engineering guide for deploying and operating the **DEEPTRACE AI Media Forensics** system in production environments. DEEPTRACE runs as a high-performance, single-node application combining a modern React/TypeScript frontend with a FastAPI backend powered by the sovereign DeepfakeBench Xception detector, MesoNet-4 comparison detector, and OpenCV YuNet face detector.

---

## 1. System Architecture

```
                    +------------------------------------+
                    |        Client Web Browser          |
                    +-----------------+------------------+
                                      |
                       HTTP / HTTPS   |  Port 8000 / 5173
                                      v
          +------------------------------------------------------+
          |           DEEPTRACE ASGI Application Host            |
          |                                                      |
          |  +-------------------------+                         |
          |  |  React / Vite Frontend  |                         |
          |  |  (Static Bundle / SPA)  |                         |
          |  +------------+------------+                         |
          |               |                                      |
          |               | REST JSON Ingestion / WebSockets     |
          |               v                                      |
          |  +-------------------------+                         |
          |  |   FastAPI ASGI Server   |                         |
          |  |   (Uvicorn on Port 8000)|                         |
          |  +------------+------------+                         |
          |               |                                      |
          |               +-----> [Security Guards: Magic Bytes, Size, MIME]
          |               |                                      |
          |               +-----> [OpenCV YuNet Face Detection]  |
          |               |                                      |
          |               +-----> [DeepfakeBench Xception Model] |
          |               |                                      |
          |               +-----> [MesoNet-4 Comparison Model]   |
          |               |                                      |
          |               +-----> [Grad-CAM Visual Backprop]     |
          |               |                                      |
          |               +-----> [2D FFT Spectral Forensics]    |
          |               |                                      |
          |               +-----> [Temporal Stability Engine]    |
          |               |                                      |
          |               +-----> [Cryptographic SHA-256 Digest] |
          +---------------+--------------------------------------+
                          |
                          v
         +------------------------------------+
         |   Local Persistent Storage Mount   |
         |   - uploads/ (Media & Heatmaps)    |
         |   - results/ (Case JSON Snapshots) |
         |   - evaluation/ (Benchmark Data)   |
         +------------------------------------+
```

---

## 2. Requirements & Prerequisites

| Component | Minimum Version | Recommended Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10+ | 3.12 | Backend runtime, PyTorch inference, OpenCV |
| **Node.js** | 18+ LTS | 20+ LTS | Frontend asset compilation and Vite bundling |
| **NPM** | 9+ | 10+ | Node package management |
| **FFmpeg** | 4.4+ | 6.0+ | Video container demuxing and frame extraction |
| **Host RAM** | 4 GB | 8 GB+ | Model weights, PyTorch kernels, 50-face arrays |
| **Disk Space** | 2 GB | 10 GB+ | Checkpoints (100 MB), uploads, forensic artifacts |
| **CUDA (Optional)** | 11.8+ / 12.1+ | 12.4 | Optional GPU hardware acceleration (CPU default) |

---

## 3. Python Environment

The backend requires a 64-bit Python 3.10+ runtime (3.12 recommended).
- **Virtual Environment**: Maintain an isolated virtualenv under `backend/.venv`.
- **Runtime Dependencies**: Pinned in `backend/requirements.txt`:
  - `fastapi`, `uvicorn`: ASGI server and API framework
  - `torch`, `torchvision`: Deep learning runtime for DeepfakeBench Xception and MesoNet-4
  - `opencv-python-headless`: YuNet face detection and video frame decoding
  - `pillow`: Image validation and format normalization
  - `pydantic`: Schema validation and serialization
  - `scipy`, `numpy`: Numerical analysis and FFT signal processing

Installation command:
```bash
cd backend
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Node Environment

The frontend is built using Node.js 18+ LTS and React 19 with Vite.
- Package Manager: `npm` (9+ or 10+)
- Dependencies: Pinned in `package-lock.json`
- Framework: React 19, TypeScript 5.8+, Lucide React icons, Tailwind CSS / Vanilla CSS tokens
- Production Bundler: Vite 8.3+

Installation command:
```bash
npm ci
```

---

## 5. Machine Learning Checkpoints

DEEPTRACE operates with three fixed, frozen neural model files loaded as singletons:

1. **Primary Detector: DeepfakeBench Xception**
   - **Filename**: `xception_best.pth` (87.7 MB)
   - **Configuration**: `xception.yaml`
   - **Architecture**: XceptionNet with modified binary classification head
   - **Location**: `deepfake_model/DeepfakeBench/training/weights/xception_best.pth`
   - **Parameters**: 21.9 Million
   - **Input Resolution**: 256x256 RGB normalized to `[-1.0, 1.0]`
   - **Runtime**: PyTorch CPU / CUDA
2. **Comparison Detector: DeepfakeBench MesoNet-4**
   - **Filename**: `meso4_best.pth` (118 KB)
   - **Location**: `deepfake_model/DeepfakeBench/training/weights/meso4_best.pth`
   - **Parameters**: 27.9 Thousand
   - **Input Resolution**: 256x256 RGB normalized to `[0.0, 1.0]`
   - **Runtime**: PyTorch CPU / CUDA
3. **Face Detector: OpenCV YuNet**
   - **Filename**: `face_detection_yunet.onnx` (232 KB)
   - **Location**: `backend/app/models/face_detection_yunet.onnx`
   - **Runtime**: OpenCV DNN Engine

*Strict Deployment Rule*: No model training, checkpoint conversion, or dynamic weight downloads occur during user requests. All models are pre-warmed once at application startup.

---

## 6. Directory Structure

```
DEEPTRACE/
├── .env.example                 # Production configuration template
├── .gitignore                   # Excludes secrets, uploads, virtual environments
├── Dockerfile                   # Single-node multi-stage container specification
├── .dockerignore                # Container build context exclusions
├── package.json                 # Frontend dependencies & build scripts
├── vite.config.ts               # Vite bundler & reverse proxy configuration
├── dist/                        # Production frontend build artifact
├── src/                         # React / TypeScript application source code
├── deepfake_model/              # DeepfakeBench sub-repository
│   └── DeepfakeBench/
│       └── training/
│           ├── config/detector/ # xception.yaml configuration
│           └── weights/         # xception_best.pth & meso4_best.pth
└── backend/                     # FastAPI backend application
    ├── requirements.txt         # Production Python dependencies
    ├── app/
    │   ├── main.py              # Application entrypoint & lifespan
    │   ├── config.py            # Central environment & path configuration
    │   ├── api/                 # Endpoint routers (analysis, health, reports)
    │   ├── models/              # Pydantic schemas & YuNet ONNX weights
    │   ├── services/            # Forensic services (Xception, Meso4, Grad-CAM, FFT)
    │   └── utils/               # Sanitization, file helpers, logger
    ├── uploads/                 # Storage: evidence media, heatmaps, thumbnails
    ├── results/                 # Storage: persistent analysis records
    └── evaluation/              # Benchmark manifests, smoke tests, metrics
```

---

## 7. Environment Variables Reference

Configure environment variables via `.env` in the project root or through container orchestration:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | IP binding address for FastAPI ASGI server. |
| `PORT` | `8000` | Port for the backend service. |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000/api` | Full URL to backend API endpoint used by the frontend. |
| `VITE_MEDIA_BASE_URL` | `http://127.0.0.1:8000` | Base URL for static forensic media (heatmaps/uploads). |
| `CORS_ORIGINS` | `localhost defaults` | Comma-separated list of allowed web origins. |
| `UPLOADS_DIR` | `uploads` | Directory for evidence uploads and forensic artifacts. |
| `RESULTS_DIR` | `results` | Directory for analysis records and evaluation results. |
| `DEEPFAKE_MODEL_DIR` | `deepfake_model` | Root directory containing DeepfakeBench files. |
| `XCEPTION_WEIGHTS_PATH` | `.../weights/xception_best.pth` | Absolute or relative path to Xception weights. |
| `XCEPTION_CONFIG_PATH` | `.../config/detector/xception.yaml`| Absolute or relative path to Xception config. |
| `MESO4_CHECKPOINT_PATH` | `.../weights/meso4_best.pth` | Absolute or relative path to MesoNet-4 weights. |
| `YUNET_MODEL_PATH` | `app/models/face_detection_yunet.onnx` | Path to YuNet ONNX face detection model. |
| `DEEPFAKE_THRESHOLD` | `0.50` | Production binary decision threshold (fixed at 0.50). |
| `VIDEO_SAMPLE_INTERVAL`| `15` | Frame sampling stride (analyze every 15th frame). |
| `MAX_ANALYZED_FRAMES` | `30` | Upper bound of analyzed frames per video. |
| `MAX_IMAGE_SIZE_BYTES` | `20971520` (20 MB) | Maximum permitted image upload size in bytes. |
| `MAX_VIDEO_SIZE_BYTES` | `104857600` (100 MB) | Maximum permitted video upload size in bytes. |

---

## 8. Development Startup

For local iterative development:

1. **Start Backend**:
   ```bash
   cd backend
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
2. **Start Frontend**:
   ```bash
   npm.cmd run dev
   ```
   Access web interface at `http://localhost:5173`.

---

## 9. Production Startup Procedure

For high-reliability production deployment:

1. **Verify Environment**:
   Ensure all model checkpoints are in place and directories are writable.
2. **Launch ASGI Process**:
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```
   *Worker Recommendation*: A single ASGI worker process (`--workers 1`) is standard due to the in-process PyTorch inference lock (`_inference_lock`). For horizontal concurrency across CPU sockets, multiple single-worker container instances may be placed behind a load balancer.
3. **Execute Smoke Test**:
   ```bash
   python backend/evaluation/deployment_smoke_test.py --base-url http://127.0.0.1:8000
   ```

---

## 10. Frontend Production Build

Compile optimized production client assets:

```bash
# Set production backend API URL before building:
# e.g., export VITE_API_BASE_URL=https://api.deeptrace.example.com/api
npm run build
```
The output bundle will be generated in `dist/`:
- `dist/index.html` (~1.0 KB)
- `dist/assets/index-*.css` (~27.7 KB)
- `dist/assets/index-*.js` (~408.8 KB)

Serve `dist/` via Nginx, Cloudflare Pages, S3/CloudFront, or FastAPI static mounting.

---

## 11. CPU Deployment Mode

- **Status**: Production default, fully supported out of the box.
- **Hardware Acceleration**: Automatic multi-threaded BLAS/OpenMP operations via PyTorch and OpenCV DNN.
- **Thread Tuning**: PyTorch auto-detects physical CPU cores (`torch.set_num_threads()`).
- **Memory Footprint**: ~750 MB steady-state RSS; transient peak ~1,250 MB under heavy 50-face batch analysis.
- **Performance Characteristics**:
  - Image Ingestion & Face Detection: ~35 ms
  - Xception Primary Inference: ~62 ms
  - Grad-CAM Visual Backpropagation: ~580 ms
  - FFT Frequency Spectrum: ~5 ms
  - Total Single-Face Image Turnaround: ~680 ms

---

## 12. GPU Deployment Mode (CUDA-Enabled)

- **Status**: Supported with automatic hardware detection.
- **Behavior**: At startup, `get_xception_detector()` executes `torch.cuda.is_available()`. If a CUDA device is detected, model weights are moved to `cuda:0` with automatic float32 tensor allocation.
- **Graceful Fallback**: If CUDA is unavailable or encounters driver errors, the backend logs a diagnostic message and gracefully falls back to CPU execution without crashing or interrupting requests.
- **Zero Configuration**: No code changes or manual flags required.
- **Verification Note**: In the reference evaluation environment, no supported CUDA GPU was present (PyTorch detected `torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`). Hence, **GPU verification was not performed: no supported GPU available in this environment.** Automated CPU fallback was verified and active.

---

## 13. Storage Persistence

DEEPTRACE utilizes local filesystem persistence:
- **`backend/uploads/`**: Stores raw evidence media, generated Grad-CAM heatmaps, FFT frequency spectra, and video frame thumbnails.
- **`backend/results/`**: Stores immutable JSON snapshots for every forensic case (`analysis_id.json`).
- **`backend/evaluation/results/`**: Stores benchmark runs and confusion matrix outputs.

*Production Mounting*: In containerized environments, mount persistent volumes to `/app/backend/uploads` and `/app/backend/results` to ensure case records survive container restarts.

---

## 14. CORS & Network Configuration

Cross-Origin Resource Sharing is centrally governed by `backend/app/config.py`:
- **Development**: Permits `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`.
- **Production**: Configured via the `CORS_ORIGINS` environment variable (comma-separated origins).
- **Wildcard Policy**: Wildcard origins (`*`) are disallowed by default to prevent cross-site evidence exfiltration.

---

## 15. Security Considerations & Stage 9A Protections

All Stage 9A security controls are strictly enforced:
1. **Magic Byte Verification**: Verifies binary headers against known MIME signatures before decoding.
2. **Streaming Size Bounds**: Drops connections exceeding 20 MB (image) or 100 MB (video) in streaming chunks.
3. **Decompression Bomb Protection**: Pillow `MAX_IMAGE_PIXELS = 25,000,000` prevents memory exhaustion attacks.
4. **Path Traversal Protection**: Filenames sanitized via regex `[^a-zA-Z0-9_.-]`; directory traversal attempts (`../`) are blocked and scrubbed.
5. **HTML Sanitization**: All report fields are sanitized through Python `html.escape()` preventing stored XSS injection.
6. **Benchmark Root Confinement**: Dataset and manifest paths are strictly verified to reside within approved directories.
7. **Path Scrubbing in Error Responses**: Internal server paths (e.g. `C:\...` or `/home/...`) are scrubbed before returning errors to clients.

---

## 16. Health & Readiness Probes

DEEPTRACE provides two dedicated monitoring endpoints designed for container orchestrators and load balancers:

### A. Liveness Probe: `GET /api/health`
- **Purpose**: Verifies that the FastAPI process is responsive.
- **Response**: HTTP 200
  ```json
  {
    "status": "ok",
    "service": "DEEPTRACE backend",
    "version": "1.0.0"
  }
  ```

### B. Readiness Probe: `GET /api/status`
- **Purpose**: Verifies model weight availability and storage writability.
- **Security Guard**: Does **not** leak internal server paths, file locations, or environment secrets.
- **Response (Operational)**: HTTP 200
  ```json
  {
    "status": "ready",
    "service": "DEEPTRACE Media Forensics API",
    "version": "1.0.0",
    "device": "cpu",
    "models": {
      "xception_primary": "ready",
      "meso4_comparison": "ready",
      "yunet_face_detector": "ready"
    },
    "storage": {
      "uploads": "writable",
      "results": "writable"
    }
  }
  ```
- **Response (Degraded)**: HTTP 200 (`status: "degraded"` if comparison detector is missing but primary Xception is ready).
- **Response (Unavailable)**: HTTP 503 (`status: "unavailable"` if primary Xception or YuNet checkpoint is missing or storage is unwritable).

---

## 17. Troubleshooting Guide

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **Port 8000 already in use** | Stale uvicorn or python process running | Identify PID via `Get-NetTCPConnection` or `lsof -i :8000` and terminate it. |
| **`missing_checkpoint` on `/api/status`** | Model weights missing or path variable unset | Verify `deepfake_model/DeepfakeBench/training/weights/xception_best.pth` exists. |
| **HTTP 413 on upload** | Upload exceeds 20 MB (image) or 100 MB (video) | Adjust `MAX_IMAGE_SIZE_BYTES` or `MAX_VIDEO_SIZE_BYTES` in `.env`. |
| **HTTP 400 'File content does not match'** | Spoofed file extension (magic bytes mismatch) | Upload valid JPEG, PNG, WebP, MP4, or MOV media. |
| **Grad-CAM visual backprop slow** | CPU inference on very high resolution images | Normal behavior on CPU; ensure server has at least 4 physical CPU cores. |

---

## 18. Known Operational Limitations

1. **CPU Inference Latency**: On CPU-only deployments, Grad-CAM backpropagation on high-resolution images (~580 ms) and multi-face images (~7s for 50 faces) requires noticeable wall-clock compute time.
2. **Reverse Proxy Requirement**: In internet-facing deployments, DEEPTRACE should be placed behind a hardened reverse proxy (e.g. Nginx, Cloudflare, Traefik) to provide SSL/TLS termination, rate limiting, and HTTP request buffering.
3. **No Dynamic Model Switching**: The primary detector architecture is fixed to DeepfakeBench Xception at startup; switching requires server restart.

---

## 19. Single-Node Architecture Limitations

1. **Local Filesystem & JSON Storage**: DEEPTRACE stores case history, evidence hashes, and forensic reports in local JSON files and persistent filesystem directories (`uploads/` and `results/`).
2. **Not Horizontally Scalable Out-of-the-Box**: Local JSON and disk storage is **not equivalent to a distributed, horizontally scalable production database** (e.g., PostgreSQL or MongoDB cluster). Running multiple stateless replicas without a shared network filesystem (NFS/EFS) will result in split-brain case records.
3. **In-Process Locks**: Video frame and batch processing uses an in-process thread lock (`_inference_lock`) to protect PyTorch hooks, designed specifically for predictable single-node resource utilization.

---

## 20. Containerization (Docker)

A production-grade, multi-stage `Dockerfile` is included in the project root:

```bash
# Build the container image
docker build -t deeptrace:latest .

# Run the container with persistent storage volume
docker run -d \
  --name deeptrace \
  -p 8000:8000 \
  -v deeptrace_uploads:/app/backend/uploads \
  -v deeptrace_results:/app/backend/results \
  --restart unless-stopped \
  deeptrace:latest
```

*Verification Note*: In the reference evaluation environment, Docker CLI / daemon was not installed. Hence, **Docker verification was not performed: Docker CLI / daemon is not installed in this environment.** The multi-stage `Dockerfile` and `.dockerignore` were inspected and verified for correct layer definitions, pinned base images, non-root paths, and healthchecks, but active image build and container runtime tests remain unexecuted.
