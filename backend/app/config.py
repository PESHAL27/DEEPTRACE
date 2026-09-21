from pathlib import Path

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Security & Validation Constraints
import os
from PIL import Image

# Media-specific size limits (configurable via env)
MAX_IMAGE_SIZE_BYTES: int = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(20 * 1024 * 1024)))  # 20 MB
MAX_VIDEO_SIZE_BYTES: int = int(os.getenv("MAX_VIDEO_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB
MAX_FILE_SIZE_BYTES: int = MAX_VIDEO_SIZE_BYTES  # Backward compatibility alias

# Image dimension and pixel constraints (decompression bomb protection)
MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))  # 25M pixels
MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "8192"))  # 8192px width/height

# Prevent Pillow decompression bombs globally
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Video decoding resource constraints
MAX_VIDEO_DIMENSION: int = int(os.getenv("MAX_VIDEO_DIMENSION", "4096"))  # 4096px width/height
MAX_VIDEO_DURATION_SECONDS: float = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "600.0"))  # 10 minutes
MAX_VIDEO_FPS: float = float(os.getenv("MAX_VIDEO_FPS", "120.0"))  # 120 FPS

# Benchmark workload limit
MAX_BENCHMARK_SAMPLES: int = int(os.getenv("MAX_BENCHMARK_SAMPLES", "500"))

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/quicktime",
}

# Magic bytes headers for content verification
MAGIC_NUMBERS = {
    "jpg": [b"\xFF\xD8\xFF"],
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],  # WebP begins with RIFF....WEBP
    "mp4": [b"ftyp", b"\x00\x00\x00"],  # mp4 boxes
    "mov": [b"moov", b"mdat", b"ftyp", b"wide", b"\x00\x00\x00"],
}

# CORS settings for development
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Thumbnails directory for video frame previews
THUMBNAILS_DIR = UPLOADS_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

# Heatmaps directory for Grad-CAM explainability overlays
HEATMAPS_DIR = UPLOADS_DIR / "heatmaps"
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

# Forensics directories for frequency-domain artifacts
FORENSICS_DIR = UPLOADS_DIR / "forensics"
FREQUENCY_DIR = FORENSICS_DIR / "frequency"
FREQUENCY_DIR.mkdir(parents=True, exist_ok=True)

# Forensic Detection Threshold Configuration
# In binary cross-entropy classification, 0.5 represents the standard Bayesian decision boundary
# between authentic (class 0) and manipulated (class 1).
# Operators can configure DEEPFAKE_THRESHOLD via environment variable:
# - Lower values (e.g. 0.40): Higher sensitivity / recall for deepfake triage screening
# - Higher values (e.g. 0.65): Lower false alarm rate for high-confidence forensic flagging
DEEPFAKE_THRESHOLD: float = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))

# Video Sampling & Analysis Configuration
# VIDEO_SAMPLE_INTERVAL: Interval between sampled frames (e.g. 15 = sample every 15th frame, ~0.5s at 30 fps)
# MAX_ANALYZED_FRAMES: Maximum number of frames analyzed per video to ensure low latency
VIDEO_SAMPLE_INTERVAL: int = int(os.getenv("VIDEO_SAMPLE_INTERVAL", "15"))
MAX_ANALYZED_FRAMES: int = int(os.getenv("MAX_ANALYZED_FRAMES", "30"))

# Evaluation & Benchmark Isolation Directories
EVALUATION_DIR = BACKEND_DIR / "evaluation"
DATASETS_DIR = EVALUATION_DIR / "datasets"
MANIFESTS_DIR = EVALUATION_DIR / "manifests"
BENCHMARK_RESULTS_DIR = EVALUATION_DIR / "results"
BENCHMARK_REPORTS_DIR = EVALUATION_DIR / "reports"

for d in (EVALUATION_DIR, DATASETS_DIR, MANIFESTS_DIR, BENCHMARK_RESULTS_DIR, BENCHMARK_REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

