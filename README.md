# DEEPTRACE — AI Digital Media Forensics

**DEEPTRACE** is an end-to-end digital media forensics platform engineered for verifiable deepfake detection, visual attribution, frequency analysis, and media provenance auditing. It pairs an asynchronous FastAPI backend powered by PyTorch deep learning architectures with a modern React 19 Single Page Application (SPA).

---

## 1. System Overview & Model Disclosures

DEEPTRACE provides forensic examiners with multi-layered, explainable evidence without relying on synthetic risk scores or heuristic weighting:

- **Primary Classification**: Sovereign DeepfakeBench Xception neural network (`xception_best.pth`, 87.7 MB) evaluating localized face crops with a fixed 0.50 decision threshold. Output probabilities represent the raw, uncalibrated manipulated-class softmax score.
- **Cross-Architecture Validation**: MesoNet-4 detector (`meso4_best.pth`, 118 KB) providing independent architectural cross-checking to highlight model disagreements.
- **Face Localization**: OpenCV YuNet ONNX detector (`face_detection_yunet.onnx`, 232 KB) supporting multi-face extraction with normalized coordinate bounding boxes.
- **Explainable Attribution**: Layer-specific Grad-CAM (Target: `conv4`) producing gradient-weighted activation heatmaps over suspected facial regions. *Grad-CAM visualizes model attention focus; it does not delineate ground-truth manipulation boundaries.*
- **Spectral Forensics**: 2D Fast Fourier Transform (FFT) generating azimuthally integrated radial power spectrum profiles to detect frequency-domain synthesis artifacts. *Frequency findings are physical signal observations, not independent proof of tampering.*
- **Temporal Stability**: Frame-by-frame manipulation variance and optical flow analysis across sampled video sequences. Anomaly spikes flag transient flicker or warping.
- **Evidence Integrity**: SHA-256 cryptographic digests generated at ingestion, immutable report records, and printable HTML forensic exports with mandatory automated-analysis disclaimers.
- **Zero-Face Handling**: Media with no detected faces gracefully returns `"no_faces_detected"` without fabricating probabilities.

---

## 2. Quick Start & Reproducible Setup

### Prerequisites
- **Python**: 3.10+ (64-bit; 3.12 recommended)
- **Node.js**: 18+ LTS (20+ recommended) & npm 9+
- **FFmpeg**: 4.4+ (required for video container parsing; must be in system PATH)

### 1. Environment Configuration
```bash
cp .env.example .env
```
Default parameters in `.env` are pre-configured for local development.

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Start ASGI server (Uvicorn on port 8000)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Endpoints:
- API Base: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health Probe: `http://127.0.0.1:8000/api/health`
- Readiness Probe: `http://127.0.0.1:8000/api/status`

### 3. Frontend Setup
```bash
# From repository root
npm install

# Start Vite development server (Port 5173)
npm run dev

# Compile production bundle
npm run build
```
Access the client web application at `http://localhost:5173`.

---

## 3. Reviewer Demonstration Guide (5-Minute Walkthrough)

To verify full system capabilities using included deterministic fixtures:

1. **System Health Check**: Visit `http://localhost:5173/`. Verify active badges for Xception, Meso4, and YuNet on the Dashboard.
2. **Authentic Image**: Navigate to **New Analysis**, upload `deepfake_model/tests/samples/known_real.jpg`. Verdict: `Likely authentic` (~0.62 model probability; Xception and Meso4 agree).
3. **Manipulated Image**: Upload `deepfake_model/tests/samples/known_fake.jpg`. Verdict: `Likely manipulated` (~0.81 – 0.98 probability). Inspect Grad-CAM activation heatmap and 2D FFT spectrum.
4. **Multi-Face Crowd Batch**: Upload `deepfake_model/tests/samples/all_in_one.jpg`. 50 discrete faces are boxed and analyzed simultaneously on CPU (~5.3s).
5. **Video Temporal Tracking**: Upload `backend/test_fixtures/test_fake_video.mp4`. 8 sampled frames analyzed; temporal consistency graph highlights the anomaly spike at Frame 4. Scrubber switches frame heatmaps.
6. **Forensic Report Export**: Open report modal for any case, verify cryptographic SHA-256 digest, and click **Printable Report (HTML)** for the clean export document.

*See [HANDOFF.md](HANDOFF.md) for the complete academic demonstration script.*

---

## 4. Architecture & Directory Structure

```
DEEPTRACE/
├── backend/
│   ├── app/                 # FastAPI application routes, schemas, and forensic services
│   ├── deepfake_model/      # Pretrained weights (Xception, Meso4, YuNet)
│   ├── evaluation/          # Benchmarks, manifests, validation scripts, and reports
│   ├── test_fixtures/       # Deterministic image and video test media
│   ├── uploads/             # Ephemeral upload storage & generated heatmaps
│   └── results/             # Persistent JSON analysis snapshots
├── src/                     # React 19 / TypeScript SPA source code
├── dist/                    # Compiled production assets
├── Dockerfile               # Production container definition
├── HANDOFF.md               # Academic demonstration & handoff guide
├── FINAL_TEST_REPORT.md     # Stage 9D comprehensive validation report
├── RELEASE_CHECKLIST.md     # Pre-release audit checklist
└── README.md                # This documentation
```

---

## 5. Execution Environment & Concurrency Model

- **CPU Runtime Mode**: Fully tested, validated, and operational. All 152 regression tests, inference pipelines, and multi-face batch evaluations pass on host CPU.
- **Concurrency & Inference Serialization**: FastAPI handles incoming HTTP requests asynchronously. Deep learning forward passes and Grad-CAM backprop are serialized via an internal threading lock (`_inference_lock`) to guarantee thread safety for PyTorch model singletons. Concurrent requests queue cleanly without race conditions.
- **Uncalibrated Model Probability**: Model probabilities reflect the raw manipulated-class softmax output of the neural network. No empirical post-hoc probability calibration (e.g., Platt scaling or temperature scaling) is claimed or applied.
- **GPU Acceleration**: *GPU verification not performed: no supported GPU available in this environment.* The codebase includes automated CPU fallback.
- **Docker Containerization**: *Docker verification not performed: Docker CLI / daemon is not installed in this environment.* Container configuration is provided in `Dockerfile` and `docker-compose.yml`.
- **Single-Node Storage**: Case records and uploaded artifacts are stored on the local filesystem (`results/` and `uploads/`), requiring volume persistence when containerized.

---

## 6. Verification & Automated Testing

To execute the test suites:

```bash
# Full backend regression suite (152 unique tests)
python -m unittest discover -s backend -p "test_*.py"

# Production E2E pipeline
python backend/evaluation/test_production_e2e.py

# Deployment smoke test (requires running backend)
python backend/evaluation/deployment_smoke_test.py --base-url http://127.0.0.1:8000

# Cross-system consistency audit
python backend/evaluation/test_consistency_audit.py

# Frontend production build
npm run build
```

---

## 7. Troubleshooting & Recovery

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: Xception weights checkpoint not found` | Checkpoint missing from `training/weights/` | Ensure weights exist at `deepfake_model/DeepfakeBench/training/weights/xception_best.pth` or configure `XCEPTION_WEIGHTS_PATH` in `.env`. |
| `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` | FFmpeg not in PATH | Install FFmpeg via package manager (`winget install Gyan.FFmpeg`, `apt install ffmpeg`, or `brew install ffmpeg`) and restart terminal. |
| `OSError: [Errno 10048] address already in use` | Port 8000 or 5173 occupied | Terminate the process using port 8000 or configure `PORT=8001` in `.env` and update `VITE_API_BASE_URL`. |
| `ModuleNotFoundError: No module named 'app'` | Command run outside `backend/` | Execute Uvicorn from inside the `backend/` directory or add `backend/` to `PYTHONPATH`. |

---

## 8. Authoritative Documents

- [HANDOFF.md](HANDOFF.md) — Academic handoff guide and demonstration walkthrough.
- [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md) — Audited test records and pass/fail counts.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) — Pre-release validation sign-off checklist.
- [DEPLOYMENT.md](backend/evaluation/DEPLOYMENT.md) — Production operations and configuration guide.
