from pathlib import Path
import os
from PIL import Image

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    _backend_dir = Path(__file__).resolve().parent.parent
    _root_dir = _backend_dir.parent
    if (_root_dir / ".env").exists():
        load_dotenv(_root_dir / ".env")
    elif (_backend_dir / ".env").exists():
        load_dotenv(_backend_dir / ".env")
    else:
        load_dotenv()
except ImportError:
    pass

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BACKEND_DIR.parent

# Configurable storage directories (supports relative to BACKEND_DIR or absolute)
_uploads_env = os.getenv("UPLOADS_DIR", "uploads")
UPLOADS_DIR = Path(_uploads_env) if os.path.isabs(_uploads_env) else (BACKEND_DIR / _uploads_env)

_results_env = os.getenv("RESULTS_DIR", "results")
RESULTS_DIR = Path(_results_env) if os.path.isabs(_results_env) else (BACKEND_DIR / _results_env)

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# Server binding configuration
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Configurable Model Paths
DEEPFAKE_MODEL_DIR = Path(os.getenv("DEEPFAKE_MODEL_DIR", str(WORKSPACE_ROOT / "deepfake_model")))
BENCH_DIR = DEEPFAKE_MODEL_DIR / "DeepfakeBench"
TRAINING_DIR = BENCH_DIR / "training"

XCEPTION_CONFIG_PATH = Path(os.getenv("XCEPTION_CONFIG_PATH", str(TRAINING_DIR / "config" / "detector" / "xception.yaml")))
XCEPTION_WEIGHTS_PATH = Path(os.getenv("XCEPTION_WEIGHTS_PATH", str(TRAINING_DIR / "weights" / "xception_best.pth")))
MESO4_WEIGHTS_PATH = Path(os.getenv("MESO4_CHECKPOINT_PATH", str(TRAINING_DIR / "weights" / "meso4_best.pth")))
YUNET_MODEL_PATH = Path(os.getenv("YUNET_MODEL_PATH", str(BACKEND_DIR / "app" / "models" / "face_detection_yunet.onnx")))

# CORS settings (configurable via comma-separated CORS_ORIGINS)
_cors_env = os.getenv("CORS_ORIGINS", "")
if _cors_env.strip():
    CORS_ORIGINS = [orig.strip() for orig in _cors_env.split(",") if orig.strip()]
else:
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

