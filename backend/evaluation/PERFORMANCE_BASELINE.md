# DEEPTRACE — Performance & Efficiency Baseline (Stage 9B)

## Executive Summary
This document records the empirical pre-optimization baseline for DEEPTRACE Media Forensics across 8 core operational workloads. Measurements were executed against the live production FastAPI backend (`http://127.0.0.1:8000`) on the host processor with continuous 5 ms resident set size (RSS) memory sampling via `psutil`.

---

## 1. Hardware & Execution Environment
- **Host CPU**: Multi-core x86_64 host processor (6 physical cores / 12 logical threads detected)
- **PyTorch Threads**: 6 worker threads (`torch.get_num_threads() = 6`)
- **OpenCV Threads**: 12 worker threads (`cv2.getNumThreads() = 12`)
- **GPU Acceleration**: `N/A` (Inference executes via PyTorch CPU multi-threaded kernels)
- **Host OS**: Windows 11
- **Python Environment**: Python 3.12 (Isolated `.venv`)

---

## 2. Stage-by-Stage Pipeline Profile Breakdown

Detailed latency breakdown measured via high-resolution `time.perf_counter()` on representative fixtures:

### A. Still Image Pipeline Breakdown (`known_real.jpg`, 1200x896, 2 faces)
| Pipeline Stage | Wall-Clock Latency | Share of Total | Description |
| :--- | :--- | :--- | :--- |
| **Image Decode** | `18.85 ms` | 1.7% | `cv2.imread` BGR buffer decode from disk |
| **Metadata Extraction** | `6.13 ms` | 0.5% | Pillow EXIF header + resolution extraction |
| **Face Detection (YuNet)** | `109.85 ms` | 9.8% | OpenCV YuNet ONNX inference (detects 2 faces) |
| **Xception Inference** | `337.40 ms` | 30.1% | Pretrained Xception forward pass (168.7 ms/face) |
| **Meso4 Comparison** | `29.42 ms` | 2.6% | MesoNet-4 independent forward pass (14.7 ms/face) |
| **Grad-CAM Attribution** | `580.31 ms` | 51.8% | Conv4 activation backpropagation + heatmap generation |
| **FFT Spectral Forensics**| `29.50 ms` | 2.6% | 2D FFT, radial energy distribution, high-freq ratio |
| **Report Generation** | `9.38 ms` | 0.8% | Cryptographic hashing + JSON report construction |
| **Total Ingestion Pipeline**| **`1,120.84 ms`** | **100.0%** | Full forensic analysis pipeline |

### B. Video Pipeline Breakdown (`test_fake_video.mp4`, 318x318, 30 frames, 1s @ 30 FPS)
| Pipeline Stage | Wall-Clock Latency | Share of Total | Description |
| :--- | :--- | :--- | :--- |
| **Video Open & Telemetry**| `47.98 ms` | 4.3% | Container header inspection, FPS/duration read |
| **Frame Decoding** | `18.89 ms` | 1.7% | Stride-15 frame seek and read (2 sampled frames) |
| **Face Detection** | `15.76 ms` | 1.4% | YuNet detection across 2 sampled frames |
| **Xception Inference** | `393.86 ms` | 35.6% | Per-frame face crop inference (196.9 ms/frame) |
| **Meso4 Comparison** | `22.92 ms` | 2.1% | Per-frame Meso4 inference (11.5 ms/frame) |
| **Grad-CAM Attribution** | `585.97 ms` | 53.0% | Conv4 visual attribution on top suspicious frame |
| **FFT Spectral Forensics**| `19.57 ms` | 1.8% | Frequency spectrum on top suspicious frame |
| **Temporal Consistency** | `1.48 ms` | 0.1% | Consecutive frame probability deltas, volatility dispersion, bbox trajectory tracking, and temporal anomaly events |
| **Total Video Pipeline** | **`1,106.44 ms`** | **100.0%** | Complete video forensic evaluation |

---

## 3. Pre-Optimization Workload Benchmark (Scenarios A through H)

Measurements recorded against live FastAPI server (PID 11128):

| Scenario | Fixture & Workload | Latency (Wall-Clock) | Process RSS Before | Peak RSS (Sampled) | Process RSS After | Analyzed Entities | GPU VRAM | Response Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Normal Image** | `known_real.jpg`<br>(1200x896, 2 faces) | `0.758 s` | `348.95 MB` | `545.70 MB` | `501.04 MB` | 2 faces, 1 image | `N/A` | `5,386 B` |
| **B. Normal Video** | `test_fake_video.mp4`<br>(318x318, 30 fps, 1s) | `0.997 s` | `501.04 MB` | `553.78 MB` | `489.90 MB` | 2 frames analyzed, 1 face | `N/A` | `15,452 B` |
| **C. Multi-Face Image** | `all_in_one.jpg`<br>(2570x2570, 50 faces) | `7.773 s` | `489.90 MB` | `968.19 MB` | `913.08 MB` | 50 faces, 1 image | `N/A` | `23,768 B` |
| **D. Multi-Frame Video**| `test_real_video.mp4`<br>(640x480, 2 faces/fr) | `1.623 s` | `913.08 MB` | `975.19 MB` | `904.11 MB` | 2 frames analyzed, 2 faces | `N/A` | `18,106 B` |
| **E. Report Generation**| `REP-DT-2026-87687958`<br>(JSON serialization) | `0.0088 s` | `904.11 MB` | `904.57 MB` | `904.57 MB` | Case audit lookup | `N/A` | `11,147 B` |
| **F. Printable Report** | `REP-DT-2026-87687958`<br>(HTML rendering) | `0.0051 s` | `904.57 MB` | `904.65 MB` | `904.65 MB` | Standalone print view | `N/A` | `10,223 B` |
| **G. Concurrent Images** | `2x known_real.jpg`<br>(2 simultaneous workers)| `1.551 s` total<br>(T1: 1.550s, T2: 0.772s) | `904.65 MB` | `931.95 MB` | `874.79 MB` | Serialized via `_inference_lock` | `N/A` | N/A |
| **H. Concurrent Videos** | `2x test_fake_video.mp4`<br>(2 simultaneous workers)| `2.696 s` total<br>(T1: 1.183s, T2: 2.691s) | `874.79 MB` | `936.27 MB` | `861.90 MB` | Serialized via `_inference_lock` | `N/A` | N/A |

---

## 4. Key Bottlenecks Identified in Audit

1. **Redundant Face Detection Calls in `/api/analyze`**:
   - `app/api/analysis.py` previously called `detect_faces(target_path, media_type)` at line 172, immediately followed by `analyze_image(target_path)` or `analyze_video(target_path)`.
   - In image analysis, this executed OpenCV YuNet face detection twice sequentially on the identical image.
   - In video analysis, `detect_faces` opened the video and sampled 10 frames, and then `analyze_video` opened the video again and sampled 30 frames. The initial `face_result` was then discarded and replaced with results from `detection_result.frame_results`.
2. **Duplicate Media File Openings for Metadata**:
   - Ingestion opened media files twice with Pillow (once in `extract_media_metadata` and once in `analyze_metadata`).
3. **Inference Mode Dispatch**:
   - `predict_face_crop` in `detection_service.py` and `meso4_adapter.py` utilized `torch.no_grad()` rather than `torch.inference_mode()`, which incurs additional C++ autograd view tracking and version counter bookkeeping.
4. **Grad-CAM Object Recreation**:
   - `generate_gradcam_for_face` instantiated a new `XceptionGradCAM()` Python wrapper on every single invocation rather than caching the engine wrapper.
5. **CPU Batching vs. Sequential Execution**:
   - Profiling batched tensor execution vs sequential execution on CPU showed that large batches (e.g. 50 faces in `all_in_one.jpg`) incurred CPU cache thrashing (Speedup: 0.94x, 8296ms batched vs 7782ms sequential). Therefore, batching on CPU is empirically counter-productive and rejected.
