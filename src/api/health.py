"""Health endpoint for system health checks."""

import structlog
from fastapi import APIRouter, HTTPException

from ..core.models import HealthResponse
from ..services.vector_store import vector_store_service

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        vector_stats = vector_store_service.get_stats()

        return HealthResponse(
            status="healthy",
            version="1.0.0"
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")
