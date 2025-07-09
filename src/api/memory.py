"""Memory endpoint for session memory management."""

import structlog
from fastapi import APIRouter, HTTPException

from ..core.models import ClearMemoryRequest, ClearMemoryResponse
from ..agents.memory_agent import memory_agent
from ..services.vector_store import vector_store_service
from ..services.pdf_processor import pdf_processor_service

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


# @router.post("/clear-vector-store", response_model=ClearMemoryResponse)
# async def clear_vector_store():
#     """Clear all documents from the vector store."""
#     try:
#         logger.info("Clearing vector store")

#         success = await vector_store_service.delete_all()

#         if success:
#             return ClearMemoryResponse(
#                 success=True,
#                 message="Vector store cleared successfully"
#             )
#         else:
#             return ClearMemoryResponse(
#                 success=False,
#                 message="Failed to clear vector store"
#             )

#     except Exception as e:
#         logger.error("Error clearing vector store", error=str(e))
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error clearing vector store: {str(e)}"
#         )


# @router.post("/reprocess-pdfs", response_model=ClearMemoryResponse)
# async def reprocess_pdfs():
#     """Clear vector store and reprocess all PDFs with enhanced table handling."""
#     try:
#         logger.info("Reprocessing PDFs with enhanced table handling")

#         # First clear the vector store
#         clear_success = await vector_store_service.delete_all()
#         if not clear_success:
#             return ClearMemoryResponse(
#                 success=False,
#                 message="Failed to clear vector store"
#             )

#         # Then reprocess all PDFs
#         total_chunks = await pdf_processor_service.process_pdf_directory(force=True)

#         if total_chunks > 0:
#             return ClearMemoryResponse(
#                 success=True,
#                 message=f"Successfully reprocessed PDFs with {total_chunks} chunks"
#             )
#         else:
#             return ClearMemoryResponse(
#                 success=False,
#                 message="No PDFs were processed"
#             )

#     except Exception as e:
#         logger.error("Error reprocessing PDFs", error=str(e))
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error reprocessing PDFs: {str(e)}"
#         )
