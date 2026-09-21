from fastapi import APIRouter
from ..models.schemas import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify backend operational readiness."""
    return HealthResponse(
        status="ok",
        service="DEEPTRACE backend",
        version="1.0.0"
    )
