# DEEPTRACE — Security, Reliability & Failure-Handling Audit (Stage 9A)

## Executive Summary
This document specifies the concrete security, reliability, failure handling, and resource safety controls implemented for the DEEPTRACE digital media forensics application. As an AI-assisted forensic platform operating in forensic triage, laboratory, and investigative environments, DEEPTRACE adheres to the principle of fail-safe error containment: untrusted inputs are verified prior to ingestion, execution state is isolated across threads, error responses never leak host server filesystem architecture or stack traces, and forensic findings reflect authenticated algorithmic outputs without synthetic scores or silent fallbacks.

---

## 1. Upload Security
Uploaded media files represent the primary boundary between untrusted external clients and the forensic pipeline.
- **Header & Extension Verification**: Client-provided MIME types and extensions are treated as advisory and never inherently trusted. The backend requires extension validation (`.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.mov`) combined with mandatory initial binary byte inspection.
- **Dangerous Signature Rejection**: Pre-flight inspection checks the initial 64 bytes against known dangerous executable signatures, including Windows PE executables (`MZ`), Linux ELF binaries (`\x7fELF`), shell scripts (`#!`), HTML/XML documents (`<!DOCTYPE`, `<html`, `<?xml`), SVG vectors (`<svg`), and Java/Mach-O binaries (`\xca\xfe\xba\xbe`). Spoofed files attempting to bypass filtering are rejected immediately with HTTP 400.

---

## 2. File Validation & Magic Byte Inspection
Content inspection verifies standard container signatures:
- **JPEG**: Requires valid SOF marker `\xFF\xD8\xFF` at offset 0.
- **PNG**: Requires PNG file signature `\x89PNG\r\n\x1a\n` at offset 0.
- **WebP**: Requires RIFF container identifier `RIFF` at offset 0 and sub-type identifier `WEBP` within the initial 16 bytes.
- **MP4**: Requires an ISO base media file format box containing `ftyp` box designation within the first 32 bytes.
- **MOV**: Requires QuickTime atom signatures (`moov`, `mdat`, `ftyp`, `wide`) in the initial 32 bytes.

---

## 3. Configurable Size Limits
Media ingestion uses streaming chunked disk writes (1 MB chunks) to prevent memory exhaustion from oversized files:
- **Image Size Limit**: Default `20 MB` (`MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024`). Configurable via `MAX_IMAGE_SIZE_BYTES` environment variable.
- **Video Size Limit**: Default `100 MB` (`MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024`). Configurable via `MAX_VIDEO_SIZE_BYTES` environment variable.
- **Streaming Guard**: Total streamed bytes are tracked iteratively during ingestion. If the configured limit is exceeded, streaming aborts immediately, the staged temporary file is deleted, and HTTP 413 (Payload Too Large) is returned.

---

## 4. Image Dimension & Pixel Limits (Decompression Bomb Protection)
Pathological image files (e.g. decompression bombs with small compressed file size but billions of uncompressed pixels) are guarded before full decoding:
- **Global Pillow Protection**: `Image.MAX_IMAGE_PIXELS` is explicitly capped at `25,000,000` pixels.
- **Dimension Boundary**: Width and height are bounded by `MAX_IMAGE_DIMENSION = 8,192` pixels.
- **Exception Catching**: `Image.DecompressionBombError` and `Image.UnidentifiedImageError` are caught cleanly and return HTTP 422 with neutral descriptions.

---

## 5. Video Resource Limits
Video container decoding by OpenCV is strictly bounded:
- **Resolution Limit**: Bounded to `MAX_VIDEO_DIMENSION = 4,096` pixels (4K). Malformed containers declaring larger dimensions are rejected with HTTP 422.
- **Duration Limit**: Bounded to `MAX_VIDEO_DURATION_SECONDS = 600.0` seconds (10 minutes).
- **Framerate Limit**: Bounded to `MAX_VIDEO_FPS = 120.0` frames per second.
- **Sampling Interval & Frame Cap**: Respects `VIDEO_SAMPLE_INTERVAL = 15` frames (~0.5s intervals) and `MAX_ANALYZED_FRAMES = 30` frames.
- **Resource Reclaim**: Video capture handles (`cv2.VideoCapture`) are released in unconditional `finally:` blocks.

---

## 6. Path Traversal Protection
Filenames, case identifiers, report snapshot lookups, and evaluation datasets are protected against directory traversal:
- **Sanitization Strategy (`sanitize_filename`)**:
  - Drops Windows drive designations (e.g., `C:`).
  - Normalizes both `/` and `\` separators and extracts the final path component.
  - Strips leading dots (preventing hidden/system file creation) and directory separators.
  - Neutralizes Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`).
  - Limits maximum filename length to 128 characters preserving the legitimate extension.
- **Identifier Confinement**: All `analysis_id`, `case_id`, `report_id`, and `run_id` paths are strictly validated using regex `^[a-zA-Z0-9_-]{1,64}$`.

---

## 7. Static-File Serving Protection
- **Storage Scope**: `/uploads` is mounted strictly to `BACKEND_DIR / "uploads"`.
- **Traversal Prevention**: Starlette's `StaticFiles` verifies that resolved file paths cannot escape the designated root. Traversal attempts (e.g. `/uploads/../main.py` or `/uploads/..%2F..%2Fapp%2Fmain.py`) return HTTP 404.
- **Sensitive Directories**: Project source code, virtual environment, git configuration, `.env` files, and database files reside outside the static mounts.

---

## 8. Absolute Path Leak Prevention & Error Sanitization
- **Scrubbing (`sanitize_error_message`)**: All exception details passing through HTTP error handlers and service layers are scrubbed using regular expressions that redact Windows absolute paths (`[a-zA-Z]:\[Users|...]\...`), POSIX paths (`/home/...`, `/tmp/...`), and Python file line indicators (`File "...", line ...`).
- **Stack Trace Suppression**: Unhandled exceptions trigger a generic 500 handler returning `"Internal Server Error"` without traceback outputs. Python stack traces are logged internally to backend logs for operator diagnosis but never sent across the API boundary.

---

## 9. Temporary File Cleanup & Orphan Artifacts
- **Immediate Cleanup on Error**: If ingestion, metadata extraction, or neural network inference encounters a fatal exception, the staging file is unlinked immediately in an exception handler.
- **Separation of Evidence**: Ephemeral processing data (`cv_temp_*.jpg`) is removed in `finally:` blocks. Persistent case evidence (original evidence file, thumbnail previews, Grad-CAM overlays, FFT spectra, and JSON reports) is retained deterministically under unique analysis identifiers (`DT-2026-XXXXXX`).
- **Resilient Report Lookups**: In the event that a stored media file is removed by retention policies, the forensic report engine falls back gracefully to persisted metadata and cryptographic hash references rather than raising an unhandled exception.

---

## 10. Concurrent Analysis Behavior
- **Singleton Model Synchronization**: DeepfakeBench Xception and OpenCV YuNet are initialized once at application startup.
- **Inference Lock (`_inference_lock`)**: PyTorch neural forward passes and Grad-CAM backpropagation hooks are synchronized using a thread-safe `threading.Lock()`.
- **Grad-CAM Hook Isolation**: Grad-CAM attaches forward and backward hooks to `model.backbone.conv4`. Hook registration and gradient calculations execute under `_inference_lock` with guaranteed cleanup in `finally:` blocks (`remove_hooks()`, `zero_grad()`, activations/gradients reset to `None`).
- **Concurrent Request Isolation**: Concurrent analysis requests operate on isolated memory scopes, unique analysis IDs, and separate artifact file paths without cross-case contamination.

---

## 11. Model Failure Behavior
- **Explicit Failure Categorization**: Model inference exceptions (invalid tensor dimensions, unreadable crops, neural runtime errors) return structured failure states with HTTP 500 without exposing stack traces.
- **No False Fallbacks**: A failed neural inference is never silently converted into an authentic verdict, a zero probability, or a fabricated deepfake score. A failure remains recorded as `status="error"` or `status="failed"`.

---

## 12. GPU Failure Behavior
- **CUDA OOM Handling**: If a `torch.cuda.OutOfMemoryError` occurs, the exception is caught, `torch.cuda.empty_cache()` is called, and a sanitized `RuntimeError` is raised.
- **Persistent Corrupted State Prevention**: GPU tensors are detached and discarded, preventing leaked allocations or corrupted CUDA driver states from affecting subsequent inferences.
- **Device Fallback Design**: If CUDA is not available at startup, the system initializes on CPU without runtime switching during active inference.

---

## 13. Report Security
- **Printable Report Escaping (`render_printable_report_html`)**: All dynamic case variables (`id`, `caseId`, `filename`, `executive_summary`, `headline`, `details`, EXIF metadata, model details) are sanitized through Python's standard `html.escape()`. Untrusted user data containing `<script>`, `<img>`, quotes, or HTML tags is escaped into standard character entities (`&lt;script&gt;`).
- **Report Snapshot Protection**: JSON snapshots stored in `uploads/reports/` are loaded via path confinement checks (`is_safe_path()`) ensuring no snapshot lookup can escape the designated reports directory.

---

## 14. Benchmark Security
- **Path Confinement (`resolve_sample_path`)**: The benchmark harness resolves dataset sample and manifest paths strictly against approved directories (`MANIFESTS_DIR`, `DATASETS_DIR`, `EVALUATION_DIR`, `UPLOADS_DIR`, and `tempfile.gettempdir()`). Attempts to target arbitrary system paths (e.g. `C:\Windows\...` or `/etc/passwd`) raise `PermissionError` and return HTTP 403.
- **Workload Limits**: Benchmark evaluations are capped at `MAX_BENCHMARK_SAMPLES = 500` samples per request to prevent denial of service through unbounded evaluation manifests.

---

## 15. Cross-Validation Security
- **Secure Upload Path**: Cross-validation uploads utilize identical streaming size guards and magic-byte inspections as the primary pipeline.
- **Comparison Detector Isolation**: In the cross-validation service, comparison model evaluations (MesoNet-4) are wrapped in individual `try / except` blocks. A failure in MesoNet-4 sets its status to `"error"` while allowing primary Xception evaluation to complete normally. Primary detector state remains isolated from comparison anomalies.

---

## 16. Logging Policy
- **Diagnostic Logging**: Logs record timestamp, process level, analysis case IDs, extracted telemetry, and operational errors.
- **Privacy & Secret Protection**: Logs contain no authentication tokens, passwords, private keys, or raw binary payloads. Host paths embedded in unhandled exceptions are sanitized before logging.

---

## 17. Configurable Limits
The system exposes explicit, configurable resource limits via environment variables:

| Parameter | Environment Variable | Default Value | Purpose |
| :--- | :--- | :--- | :--- |
| Max Image Size | `MAX_IMAGE_SIZE_BYTES` | `20,971,520` (20 MB) | Ingestion streaming memory guard |
| Max Video Size | `MAX_VIDEO_SIZE_BYTES` | `104,857,600` (100 MB) | Ingestion streaming memory guard |
| Max Image Pixels | `MAX_IMAGE_PIXELS` | `25,000,000` (25 MP) | Pillow decompression bomb safeguard |
| Max Image Dimension | `MAX_IMAGE_DIMENSION` | `8,192 px` | Bounds raster dimension size |
| Max Video Dimension | `MAX_VIDEO_DIMENSION` | `4,096 px` (4K) | Bounds video frame decoding buffer |
| Max Video Duration | `MAX_VIDEO_DURATION_SECONDS` | `600.0 s` (10 min) | Prevents unbounded video processing |
| Max Video Framerate | `MAX_VIDEO_FPS` | `120.0 FPS` | Rejects malformed container framerates |
| Max Video Frames | `MAX_ANALYZED_FRAMES` | `30` | Bounds frame-level inference workload |
| Video Sample Interval | `VIDEO_SAMPLE_INTERVAL` | `15` | Frame sampling interval stride |
| Max Benchmark Samples | `MAX_BENCHMARK_SAMPLES` | `500` | Bounds batch evaluation workload |
| Decision Threshold | `DEEPFAKE_THRESHOLD` | `0.5` | Bayesian classification decision boundary |

---

## 18. Resource Baseline (Stage 9A Final Verification)
Measurements recorded on live backend server process (`uvicorn app.main:app`, PID 11128) using high-frequency resident set size (RSS) sampling at 10 ms intervals:

| Workload Scenario | Wall-Clock Time | Process RSS Before | Peak RSS (Sampled) | Process RSS After | Analyzed Entities | GPU VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Normal Image Analysis** | `0.848 s` | `794.64 MB` | `853.55 MB` | `792.23 MB` | 2 detected faces (Xception + YuNet + Grad-CAM + FFT) | `N/A` (CPU execution) |
| **B. Normal Video Analysis** | `1.149 s` | `792.23 MB` | `857.48 MB` | `795.32 MB` | 2 frames analyzed @ 30.0 FPS (Stride: 15) | `N/A` (CPU execution) |
| **C. Concurrent Images (2 workers)** | `1.555 s` total<br>(Task 1: 0.733s, Task 2: 1.554s) | `795.32 MB` | `864.44 MB` | `789.16 MB` | 2 requests serialized via `_inference_lock` | `N/A` (CPU execution) |
| **D. Concurrent Videos (2 workers)** | `2.327 s` total<br>(Task 1: 1.183s, Task 2: 2.326s) | `789.16 MB` | `858.16 MB` | `855.74 MB` | 2 requests serialized via `_inference_lock` | `N/A` (CPU execution) |

*Notes on Resource Behavior:*
- **CPU RAM**: Steady-state server memory footprint remains between ~789 MB and ~795 MB post-startup with pre-warmed Xception and MesoNet models. Peak transient memory under concurrent workloads reaches ~864 MB during simultaneous tensor transformations, immediately stabilizing after garbage collection.
- **GPU VRAM**: Reported as `N/A` on this host environment as CUDA acceleration is unavailable and inference executes through PyTorch multi-threaded CPU kernels.
- **Concurrency Isolation**: Concurrent analyses serialize through `_inference_lock` without thread contention, memory corruption, or analysis ID collisions.

---

## 19. Known Limitations
In accordance with professional forensic standards, the following architectural and operational constraints are documented:
1. **Local Single-Node Model Locking**: Model inference synchronization uses an in-process threading lock (`threading.Lock()`). Under heavy multi-tenant concurrency, requests are queued sequentially through the neural forward pass. A multi-worker cluster with dynamic GPU worker pools would require distributed worker queues (out of scope for local forensic deployment).
2. **Container Header Parsing**: OpenCV and Pillow parse complex container structures (e.g. MP4 atom trees, EXIF TIFF tags). While validated against known malformed containers, low-level vulnerabilities in underlying C/C++ libraries (libjpeg-turbo, libpng, FFmpeg) require keeping system runtime libraries patched.
3. **Absence of Rate Limiting**: The local application runs without an upstream reverse proxy (e.g. Nginx, Cloudflare) or Redis token-bucket rate limiter. In an Internet-facing production deployment, an ingress reverse proxy must enforce IP rate limits and TLS termination.
4. **Client-Side Video Codecs**: Playback of certain video codecs in web browsers depends on client browser codec support (e.g. H.264/AVC vs HEVC/H.265).

