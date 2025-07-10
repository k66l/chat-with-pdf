"""Memory endpoint for session memory management."""

import structlog
from fastapi import APIRouter, HTTPException

from ..core.models import ClearMemoryRequest, ClearMemoryResponse
from ..agents.memory_agent import memory_agent

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/clear-memory", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest):
    """Clear memory for a specific session."""
    try:
        logger.info("Clearing memory", session_id=request.session_id)

        success = memory_agent.clear_session(request.session_id)

        if success:
            return ClearMemoryResponse(
                success=True,
                message=f"Memory cleared for session {request.session_id}"
            )
        else:
            return ClearMemoryResponse(
                success=False,
                message=f"Session {request.session_id} not found or already empty"
            )

    except Exception as e:
        logger.error("Error clearing memory", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing memory: {str(e)}"
        )
