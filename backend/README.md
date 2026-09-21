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

## 6. Current Model Status

| Component | Architecture / Library | Current Status | Notes |
|---|---|---|---|
| **Media Extraction** | OpenCV + Pillow | **Operational** | Resolution, FPS, duration, frame counts |
| **Face Detection** | MTCNN / RetinaFace interface | **Ready for Weights** | Returns `face_detector_not_loaded` |
| **Deepfake Classifier** | PyTorch (Xception CNN) | **Ready for Weights** | Returns `model_not_loaded` (no fake predictions) |
| **Model Attribution** | Grad-CAM | **Planned** | Activates upon classifier attachment |
