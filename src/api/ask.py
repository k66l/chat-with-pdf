"""Ask endpoint for question-answering functionality."""

import uuid
import structlog
from fastapi import APIRouter, HTTPException

from ..core.models import QuestionRequest, QuestionResponse
from ..agents.orchestrator import orchestrator
from ..agents.memory_agent import memory_agent

logger = structlog.get_logger(__name__)

router = APIRouter()


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{uuid.uuid4().hex[:12]}"


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the multi-agent system."""
    try:
        # Handle session ID logic
        if request.session_id:
            # Use provided session ID
            session_id = request.session_id
            is_new_session = False

            # Check if session exists
            session_stats = memory_agent.get_session_stats(session_id)
            if not session_stats['exists']:
                # Session ID provided but doesn't exist - treat as new
                is_new_session = True
                logger.info(
                    "Session ID provided but doesn't exist, treating as new session",
                    provided_session_id=session_id
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
            # sources=response.sources,
            query_type=response.query_type,
            confidence=response.confidence,
            is_new_session=is_new_session,
            message_count=message_count
        )

        return enhanced_response

    except Exception as e:
        logger.error("Error processing question", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
