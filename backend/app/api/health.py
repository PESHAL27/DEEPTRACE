import os
import torch
from fastapi import APIRouter, Response, status
from ..models.schemas import HealthResponse, StatusResponse, ModelReadinessInfo, StorageReadinessInfo
from ..config import (
    BACKEND_DIR,
    XCEPTION_WEIGHTS_PATH,
    XCEPTION_CONFIG_PATH,
    MESO4_WEIGHTS_PATH,
    YUNET_MODEL_PATH,
    UPLOADS_DIR,
    RESULTS_DIR
)
from ..utils.logger import logger

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness check endpoint to verify backend process is active."""
    return HealthResponse(
        status="ok",
        service="DEEPTRACE backend",
        version="1.0.0"
    )

@router.get("/status", response_model=StatusResponse)
async def readiness_check(response: Response):
    """
    Readiness check endpoint to verify ML models and storage availability.
    Provides diagnostic status for deployment environments without leaking local paths or secrets.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    xception_ready = XCEPTION_WEIGHTS_PATH.exists() and XCEPTION_CONFIG_PATH.exists()
    meso4_ready = MESO4_WEIGHTS_PATH.exists()
    yunet_ready = YUNET_MODEL_PATH.exists()

    uploads_writable = os.access(UPLOADS_DIR, os.W_OK)
    results_writable = os.access(RESULTS_DIR, os.W_OK)

    if xception_ready and yunet_ready and uploads_writable:
        if meso4_ready:
            overall_status = "ready"
        else:
            overall_status = "degraded"
    else:
        overall_status = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return StatusResponse(
        status=overall_status,
        service="DEEPTRACE Media Forensics API",
        version="1.0.0",
        device=device_type,
        models=ModelReadinessInfo(
            xception_primary="ready" if xception_ready else "missing_checkpoint",
            meso4_comparison="ready" if meso4_ready else "missing_checkpoint",
            yunet_face_detector="ready" if yunet_ready else "missing_model",
            smaug_ai_detector="ready" if (BACKEND_DIR / "weights" / "smaug" / "config.json").exists() else "available"
        ),
        storage=StorageReadinessInfo(
            uploads="writable" if uploads_writable else "unwritable",
            results="writable" if results_writable else "unwritable"
        )
    )
