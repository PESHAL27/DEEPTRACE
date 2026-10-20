# DEEPTRACE — Backend Foundation

AI-powered digital media forensics application backend built with Python, FastAPI, PyTorch, and OpenCV.

---

## 1. Prerequisites
- Python 3.12+ (64-bit)
- Windows PowerShell / Linux / macOS terminal

---

## 2. Virtual Environment Setup

From the project root:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
# Upgrade pip and install pinned requirements
pip install -r requirements.txt
```

Core libraries installed:
- `fastapi` & `uvicorn`: High-performance asynchronous API framework
- `opencv-python-headless`: Video container inspection, frame decoding & metadata extraction
- `Pillow`: Image resolution and format validation
- `torch` & `torchvision`: Neural network runtime for deepfake classification models
- `pydantic`: Type-safe request/response schemas

---

## 4. Running the Backend Server

Start Uvicorn with auto-reload:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Backend Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON**: `http://127.0.0.1:8000/openapi.json`

---

## 5. API Endpoints

### `GET /api/health`
Health check endpoint confirming operational readiness.

**Response:**
```json
{
  "status": "ok",
  "service": "DEEPTRACE backend",
  "version": "1.0.0"
}
```

### `POST /api/analyze`
Accepts multipart file upload (`multipart/form-data`) under the key `file`.

**Supported Formats:**
- Images: `JPG`, `JPEG`, `PNG`, `WEBP`
- Videos: `MP4`, `MOV`
- Max file size: 100 MB

**Response:**
```json
{
  "analysis_id": "DT-2026-000001",
  "filename": "sample_video.mp4",
  "media_type": "video",
  "file_size": 19293812,
  "file_size_human": "18.4 MB",
  "status": "model_not_loaded",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "MP4",
    "file_size": 19293812,
    "file_size_human": "18.4 MB",
    "fps": 30.0,
    "frame_count": 360,
    "duration": 12.0,
    "duration_formatted": "00:12"
  },
  "faces_detected": null,
  "face_detection": {
    "status": "face_detector_not_loaded",
    "faces": [],
    "count": 0
  },
  "detection": {
    "status": "model_not_loaded",
    "model": null,
    "prediction": null,
    "probability": null
  },
  "model": null,
  "prediction": null,
  "probability": null,
  "created_at": "Sep 18, 2026 • 21:50"
}
```

### `GET /api/analysis/{analysis_id}`
Retrieves existing case telemetry and stored analysis record.

---

## 6. Model & Forensic Pipeline Status

| Component | Architecture / Checkpoint | Operational Status | Function |
|---|---|---|---|
| **Media Extraction** | OpenCV + Pillow | **Operational** | Resolution, FPS, duration, frame counts, MIME validation |
| **Face Detection** | OpenCV YuNet (`face_detection_yunet.onnx`) | **Operational** | Multi-face bounding box localization and normalization |
| **Primary Classifier** | DeepfakeBench Xception (`xception_best.pth`) | **Operational** | Primary manipulation verdict and probability score |
| **Comparison Detector**| MesoNet-4 (`meso4_best.pth`) | **Operational** | Secondary cross-validation and architectural disagreement detection |
| **Visual Attribution** | Grad-CAM (Target: `conv4`) | **Operational** | Class activation heatmap overlay for localized manipulation artifacts |
| **Spectral Forensics** | 2D Fast Fourier Transform (FFT) | **Operational** | Azimuthally integrated radial power spectrum for frequency anomalies |
| **Temporal Stability** | Frame-to-frame score variance & optical flow | **Operational** | Video consistency graph, flicker detection, temporal anomalies |
| **Forensic Integrity** | Cryptographic SHA-256 Digest | **Operational** | Immutable media checksum matching and automated disclaimers |

---

## 7. Execution Environment, Concurrency & Governance

- **CPU Execution Mode**: Fully verified and operational. Supports single-threaded and multi-threaded BLAS inference.
- **Concurrency & Model Serialization**: Requests arrive asynchronously at FastAPI endpoints; PyTorch model execution and Grad-CAM backprop are serialized via `_inference_lock` (`threading.Lock()`) to guarantee thread safety.
- **Uncalibrated Output Probability**: Model probabilities represent raw manipulated-class softmax outputs and are not calibrated probabilities.
- **Model Checkpoints**:
  - Xception: `deepfake_model/DeepfakeBench/training/weights/xception_best.pth` (87.7 MB)
  - MesoNet-4: `deepfake_model/DeepfakeBench/training/weights/meso4_best.pth` (118 KB)
  - YuNet ONNX: `app/models/face_detection_yunet.onnx` (232 KB)
- **Sample Verification Commands**:
  ```bash
  # Check system readiness probe:
  curl http://127.0.0.1:8000/api/status

  # Ingest and analyze real fixture:
  curl -X POST -F "file=@../deepfake_model/tests/samples/known_real.jpg" http://127.0.0.1:8000/api/analyze

  # Ingest and analyze fake fixture:
  curl -X POST -F "file=@../deepfake_model/tests/samples/known_fake.jpg" http://127.0.0.1:8000/api/analyze
  ```
- **GPU Acceleration**: *GPU verification not performed: no supported GPU available in this environment.* Automatic CPU fallback is active.
- **Containerization**: *Docker verification not performed: Docker CLI / daemon is not installed in this environment.*
- **Detailed Documentation**: See [HANDOFF.md](../HANDOFF.md), [DEPLOYMENT.md](evaluation/DEPLOYMENT.md), and [DEPLOYMENT_READINESS_REPORT.md](evaluation/DEPLOYMENT_READINESS_REPORT.md).


