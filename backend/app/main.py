from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, UPLOADS_DIR
from .api.health import router as health_router
from .api.analysis import router as analysis_router
from .api.models import router as models_router
from .api.reports import router as reports_router
from .api.benchmark import router as benchmark_router
from .api.cross_validation import router as cross_validation_router
from .models.schemas import ErrorResponse
from .utils.logger import logger
from .utils.file_utils import sanitize_error_message

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: initialize storage, preload ML models once at startup, clean up."""
    logger.info("Initializing DEEPTRACE Media Forensics Backend Service...")
    logger.info(f"Upload storage initialized at: {UPLOADS_DIR}")

    # Pre-load face detection model and Xception detector once at startup
    try:
        from .services.face_service import get_face_detector
        from .services.detection_service import get_xception_detector
        from .services.model_registry import get_model_registry

        logger.info("Pre-warming YuNet face detector model at startup...")
        get_face_detector()
        logger.info("Pre-warming pretrained Xception deepfake detector at startup...")
        get_xception_detector()

        # Initialize model registry and comparison detectors
        registry = get_model_registry()
        for detector in registry.get_comparison_detectors():
            logger.info(f"Pre-warming comparison detector '{detector.model_id}' at startup...")
            detector.load()

        logger.info("All ML forensic models initialized and cached in memory for inference.")
    except Exception as e:
        logger.error(f"Error during ML model pre-warming at application startup: {e}", exc_info=True)

    yield
    logger.info("DEEPTRACE backend service shutting down.")


app = FastAPI(
    title="DEEPTRACE — Media Forensics API",
    description="AI-powered detection and forensic analysis of potentially manipulated images and videos.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local development with the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static uploads directory for media preview playback
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Include API routes under /api
app.include_router(health_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(benchmark_router, prefix="/api")
app.include_router(cross_validation_router, prefix="/api")

# Clean global error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    clean_detail = sanitize_error_message(str(exc.detail))
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {clean_detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="Request Failed",
            detail=clean_detail,
            code=exc.status_code
        ).model_dump()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred while processing the forensic request.",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).model_dump()
    )

@app.get("/")
async def root():
    return {
        "service": "DEEPTRACE Media Forensics",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }
