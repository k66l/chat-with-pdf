"""Ask endpoint for question-answering functionality."""

import uuid
import re
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..core.models import QuestionRequest, QuestionResponse, SessionValidationError
from ..agents.orchestrator import orchestrator
from ..agents.memory_agent import memory_agent

logger = structlog.get_logger(__name__)

router = APIRouter()


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{uuid.uuid4().hex[:12]}"


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    pattern = r"^sess_[a-f0-9]{12}$"
    return bool(re.match(pattern, session_id))


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the multi-agent system."""
    try:
        # Handle session ID validation and logic
        if request.session_id:
            # Validate session ID format
            if not validate_session_id(request.session_id):
                logger.warning("Invalid session ID format", session_id=request.session_id)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_session_id",
                        "message": f"Session ID '{request.session_id}' has invalid format. Must match pattern: sess_[12-character-hex]",
                        "valid_format": "sess_[12-character-hex]"
                    }
                )
            
            # Use provided session ID
            session_id = request.session_id
            is_new_session = False

            # Check if session exists
            session_stats = memory_agent.get_session_stats(session_id)
            if not session_stats['exists']:
                # Session ID provided but doesn't exist - return error
                logger.warning("Session ID not found", session_id=session_id)
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "session_not_found",
                        "message": f"Session '{session_id}' not found. Use POST /ask without session_id to create a new session.",
                        "valid_format": "sess_[12-character-hex]"
                    }
                )
        else:
            # Generate new session ID
            session_id = generate_session_id()
            is_new_session = True
            logger.info("Generated new session ID", session_id=session_id)

        logger.info(
            "Received question",
            session_id=session_id,
            question_length=len(request.question),
            is_new_session=is_new_session
        )

        # Process question through orchestrator
        response = await orchestrator.process_question(
            question=request.question,
            session_id=session_id
        )

        # Get updated session stats for message count
        session_stats = memory_agent.get_session_stats(session_id)
        message_count = session_stats.get('message_count', 1)

        logger.info(
            "Question processed successfully",
            session_id=session_id,
            query_type=response.query_type.value,
            is_new_session=is_new_session,
            message_count=message_count
        )

        # Create enhanced response with session management info
        enhanced_response = QuestionResponse(
            answer=response.answer,
            session_id=session_id,
            sources=response.sources,
            query_type=response.query_type,
            confidence=response.confidence,
            is_new_session=is_new_session,
            message_count=message_count
        )

        return enhanced_response

    except HTTPException:
        # Re-raise HTTPExceptions (like 404, 400) without modification
        raise
    except Exception as e:
        logger.error("Error processing question", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
