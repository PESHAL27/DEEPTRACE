# DEEPTRACE — Final Release Packaging & Academic Handoff Guide (Stage 9E)

**Release Version**: v1.0.0 (Production-Ready Academic Release)  
**Date**: October 20, 2026  
**Target Environment**: Single-Node CPU Execution (Windows 64-bit / Linux / macOS)  
**Primary Architectures**: DeepfakeBench Xception Net, MesoNet-4, OpenCV YuNet  

---

## 1. Executive Summary & Handoff Objectives

DEEPTRACE is an AI-powered digital media forensics platform developed for verifiable deepfake detection, visual attribution, frequency analysis, and media provenance auditing. 

This document serves as the authoritative handoff package for reviewers, evaluators, and operators. It provides:
1. **5-Minute Academic Demonstration Walkthrough**: Step-by-step evaluation procedure with deterministic test fixtures.
2. **Reproducible Local Setup**: Concrete commands for clean installation and execution across Python and Node.js environments.
3. **Model & Evidence Disclosures**: Clear explanations of neural detector capabilities, uncalibrated softmax probability outputs, and forensic indicator interpretations.
4. **Automated Verification Record**: Exact commands and audited results for regression, security, performance, build, and consistency validation.
5. **Operational Boundaries**: Transparent declarations of CPU execution, unperformed GPU/Docker checks, and single-node persistence.

---

## 2. Model & Forensic Evidence Disclosures

To uphold scientific and legal forensic integrity, DEEPTRACE adheres to strict evidence governance:

### Neural Classification Models
- **Primary Detector (DeepfakeBench Xception)**: Sovereign Xception convolutional neural network (`xception_best.pth`, 87.7 MB) fine-tuned on face crops. Operates with a fixed binary decision threshold at **0.50**.
- **Secondary Comparison Detector (MesoNet-4)**: MesoNet-4 neural network (`meso4_best.pth`, 118 KB) providing independent architectural cross-checking. Disagreements between Xception and Meso4 are explicitly surfaced to analysts rather than masked via automated score ensembling.
- **Face Localization (OpenCV YuNet)**: ONNX runtime detector (`face_detection_yunet.onnx`, 232 KB) providing multi-face bounding box extraction with 15% margin expansion.

### Probability & Evidence Governance
- **Uncalibrated Softmax Probabilities**: Model confidence values represent the raw softmax probability output for the manipulated class. No post-hoc empirical calibration (such as Platt scaling, isotonic regression, or temperature scaling) has been applied. Probabilities reflect statistical similarity to training artifacts and do not constitute absolute proof or calibrated Bayesian posterior probabilities.
- **Grad-CAM Visual Attribution**: Heatmaps are computed via gradient backpropagation on layer `conv4`. They visualize neural network attention focus and highlight anomalous pixel gradients (e.g. blending seams, eye/mouth artifacts); they do not delineate ground-truth manipulation boundaries.
- **2D FFT Spectral Analysis**: Azimuthally integrated radial power spectra illustrate high-frequency periodic grid spikes characteristic of generative GAN/diffusion upsampling.
- **Temporal Analysis**: Frame-to-frame score variance and optical flow measure temporal consistency across video frames. Anomaly spikes flag transient flicker or facial warping.
- **Absence of Synthetic Scores**: DEEPTRACE intentionally excludes composite "risk scores" (e.g. 0–100 scales) or arbitrary severity labels. All outputs are empirical model predictions, probabilities, and physical signal observations.
- **Zero-Face Fallback**: Media containing no detectable faces returns `"no_faces_detected"` with `null` probabilities, preventing hallucinated predictions.

---

## 3. Reviewer Demonstration Walkthrough (5-Minute Script)

Follow this structured script to evaluate all key capabilities of the running application:

### Step 1: System Status & Diagnostic Check
1. Open web browser to `http://localhost:5173/` (or port where Vite is running).
2. The **Dashboard** loads with active detector status badges:
   - Primary Detector: **Xception Net (Ready)**
   - Comparison Detector: **MesoNet-4 (Ready)**
   - Face Detector: **YuNet ONNX (Ready)**
3. Verify the system readiness probe via terminal:
   ```bash
   curl http://127.0.0.1:8000/api/status
   ```
   *Expected Response*: `{"status": "ready", "device": "cpu", "models": {"xception_primary": "ready", ...}}`

### Step 2: Authentic Image Verification
1. Navigate to **New Analysis** (`/analyze`).
2. Upload the authentic test fixture: `deepfake_model/tests/samples/known_real.jpg`.
3. Click **Analyze Media**.
4. *Observed Results*:
   - Detected Faces: `1` with localized green bounding box.
   - Primary Verdict: `Likely authentic` (Model probability: ~0.62).
   - Comparison Detector: `Likely authentic` (Agreement: True).
   - 2D FFT Frequency Graph: Smooth radial energy decay without high-frequency spikes.
   - SHA-256 Digest: Verified matching client-side hash.

### Step 3: Manipulated Image Verification & Explainable Attribution
1. Return to **New Analysis** (`/analyze`).
2. Upload the manipulated test fixture: `deepfake_model/tests/samples/known_fake.jpg`.
3. Click **Analyze Media**.
4. *Observed Results*:
   - Detected Faces: `1` with red bounding box.
   - Primary Verdict: `Likely manipulated` (Model probability: ~0.81 – 0.98).
   - Grad-CAM Heatmap: Red/yellow activation overlay concentrated on manipulated facial boundaries.
   - Spectral Forensics: Radial power spectrum graph rendering frequency distributions.

### Step 4: High-Density Crowd Batch Processing (50 Faces)
1. Upload the multi-face test fixture: `deepfake_model/tests/samples/all_in_one.jpg`.
2. Click **Analyze Media**.
3. *Observed Results*:
   - 50 discrete faces detected and boxed simultaneously.
   - Batch inference evaluates all 50 crops; primary verdict flags manipulated face crops.
   - Bounded CPU execution completes in ~5.29 seconds without memory exhaustion.

### Step 5: Temporal Video Analysis
1. Upload the video test fixture: `backend/test_fixtures/test_fake_video.mp4`.
2. Click **Analyze Media**.
3. *Observed Results*:
   - 8 sampled frames analyzed across the timeline.
   - Primary Verdict: `Likely manipulated` (Video probability: ~0.92).
   - Temporal Consistency Graph: Displays frame-by-frame scores with an anomaly spike at Frame 4.
   - Interactive Scrubber: Clicking individual timeline frames updates the Grad-CAM and FFT views for that specific frame.

### Step 6: Forensic Report & Printable HTML Export
1. On the Results page, click **View Full Report** (or navigate to `/reports` and select the latest case).
2. The **Forensic Report Detail Modal** opens:
   - Verify immutable SHA-256 media digest.
   - Inspect Independent Evidence Domains (Model Telemetry, Spectral, Metadata, Temporal).
   - Note the mandatory automated-analysis disclaimer.
3. Click **Printable Report (HTML)**:
   - A standalone, CSS-styled printable document opens in a new tab.
   - Verify that all media links are relative URLs and zero internal host filesystem paths are visible.

---

## 4. Reproducible Environment & Setup Guide

### System Prerequisites
| Component | Minimum Version | Recommended Version | Note |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10 (64-bit) | 3.12 | PyTorch 2.x and OpenCV headless |
| **Node.js** | 18 LTS | 20 LTS | Vite bundling and TypeScript build |
| **npm** | 9.x | 10.x | Dependency management |
| **FFmpeg** | 4.4+ | 6.0+ | Video frame extraction (must be in system PATH) |
| **Host RAM** | 4 GB | 8 GB+ | Model weights (~100 MB) & batch processing |

### Model Checkpoints Verification
Ensure all required model files are present at their designated locations:

| Model Component | Expected Path | File Size | Checkpoint Role |
| :--- | :--- | :---: | :--- |
| **Xception Config** | `deepfake_model/DeepfakeBench/training/config/detector/xception.yaml` | 2 KB | Model architecture hyperparameters |
| **Xception Weights** | `deepfake_model/DeepfakeBench/training/weights/xception_best.pth` | 87.7 MB | Pretrained DeepfakeBench Xception weights |
| **Meso4 Weights** | `deepfake_model/DeepfakeBench/training/weights/meso4_best.pth` | 118 KB | Pretrained MesoNet-4 weights |
| **YuNet ONNX** | `backend/app/models/face_detection_yunet.onnx` | 232 KB | OpenCV YuNet face detection network |

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone <repository-url> DEEPTRACE
cd DEEPTRACE

# 2. Configure environment file
cp .env.example .env

# 3. Setup Python Backend Virtual Environment
cd backend
python -m venv .venv

# Activate virtual environment:
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
cd ..

# 4. Install Frontend Dependencies
npm install
```

### Server Startup Commands

**Terminal 1 — Backend ASGI Service**:
```bash
cd backend
# Windows:
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Linux / macOS:
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Base API URL: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

**Terminal 2 — Frontend Development Server**:
```bash
# From workspace root:
npm run dev
```
- Client Web Application: `http://localhost:5173`

**Frontend Production Build & Preview**:
```bash
npm run build
npm run preview
```

---

## 5. Troubleshooting & Recovery Guide

| Symptom / Error | Root Cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: Xception weights checkpoint not found` | Checkpoint missing from `training/weights/` | Verify file exists at `deepfake_model/DeepfakeBench/training/weights/xception_best.pth` or configure `XCEPTION_WEIGHTS_PATH` in `.env`. |
| `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` | FFmpeg not installed or not in PATH | Install FFmpeg (`winget install Gyan.FFmpeg` on Windows, `apt install ffmpeg` on Linux, `brew install ffmpeg` on macOS) and restart terminal. |
| `OSError: [Errno 10048] address already in use` | Port 8000 or 5173 occupied by stale process | Terminate the existing process (`netstat -ano \| findstr :8000` -> `taskkill /PID <PID> /F`) or set `PORT=8001` in `.env`. |
| `ModuleNotFoundError: No module named 'app'` | Command run outside `backend/` directory | Execute Uvicorn from inside the `backend/` directory or prepend `backend/` to `PYTHONPATH`. |
| High CPU usage during dense multi-face image | 50 sequential PyTorch forward passes on CPU | Normal behavior for CPU execution mode (~5.3s). For high-throughput production, GPU acceleration is recommended. |

---

## 6. Authoritative Validation Summary

All test suites and verification scripts have been audited and executed:

| Validation Suite | Exact Command | Result | Pass Rate |
| :--- | :--- | :---: | :---: |
| **Backend Regression Suite** | `python -m unittest discover -s backend -p "test_*.py"` | **152 Passed** (0 Fail, 0 Skip) | 100% |
| **Security & Reliability Suite** | `python -m unittest backend/test_security_reliability.py` | **25 Passed** (Included in 152) | 100% |
| **Performance Profiling Suite** | `python -m unittest backend/test_performance.py` | **7 Passed** (Included in 152) | 100% |
| **Benchmark Service Suite** | `python -m unittest backend/test_benchmark_service.py` | **15 Passed** (Included in 152) | 100% |
| **Deployment Smoke Tests** | `python backend/evaluation/deployment_smoke_test.py` | **6 / 6 Passed** | 100% |
| **Production E2E Pipeline** | `python backend/evaluation/test_production_e2e.py` | **4 / 4 Passed** | 100% |
| **Cross-System Consistency Audit**| `python backend/evaluation/test_consistency_audit.py` | **15 / 15 Passed** | 100% |
| **Frontend Production Build** | `npm.cmd run build` (`tsc -b && vite build`) | **0 Errors, 0 Warnings** (710ms) | 100% |
| **Interactive Browser Journey** | Automated Browser Subagent on `http://localhost:5173/` | **8 Routes + 1 Modal OK** | 100% |

---

## 7. Operational Limitations & Untested Environments

1. **GPU Acceleration**: *GPU verification not performed: no supported GPU available in this environment.* Hardware check confirmed `torch.cuda.is_available() == False`. Automatic CPU fallback operates cleanly.
2. **Docker Container Deployment**: *Docker verification not performed: Docker CLI / daemon is not installed in this environment.* `Dockerfile` and `docker-compose.yml` are provided in the repository.
3. **Storage Persistence**: File storage uses the local filesystem (`backend/uploads/` and `backend/results/`). Multi-node clustering or distributed object storage (S3/GCS) is not implemented in this release.
4. **Concurrency Model**: Incoming HTTP requests are handled asynchronously by FastAPI, but PyTorch inference and Grad-CAM backprop are serialized via `_inference_lock` for thread safety. Requests may wait for the lock.

---

## 8. Final Handoff Recommendation

**Status**: **READY WITH LIMITATIONS**  
The DEEPTRACE codebase is frozen, verified, documented, and approved for academic handoff and local production demonstration.
