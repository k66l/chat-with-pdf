"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    PDF_SEARCH = "pdf_search"
    WEB_SEARCH = "web_search"
    AMBIGUOUS = "ambiguous"


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str = Field(...,
                      description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata")


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, description="The question to ask")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity. If not provided, a new session will be created."
    )


class OrchestratorResponse(BaseModel):
    """Internal response model for orchestrator (without session management fields)."""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(
        default=[], description="Sources used for the answer")
    query_type: QueryType = Field(..., description="Type of query processed")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Confidence score (enhanced by evaluation)")


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str = Field(..., description="The generated answer")
    session_id: str = Field(
        ..., description="Session identifier (generated if not provided in request)")
    # sources: List[str] = Field(
    #     default=[], description="Sources used for the answer")
    query_type: QueryType = Field(..., description="Type of query processed")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Confidence score (enhanced by evaluation)")
    is_new_session: bool = Field(...,
                                 description="True if this started a new session")
    message_count: int = Field(..., ge=1,
                               description="Total messages in this session")


class ClearMemoryRequest(BaseModel):
    """Request model for clearing session memory."""
    session_id: str = Field(..., description="Session ID to clear")


class ClearMemoryResponse(BaseModel):
    """Response model for memory clearing."""
    success: bool = Field(...,
                          description="Whether the operation was successful")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0", description="API version")


class AgentState(BaseModel):
    """State model for LangGraph agent orchestration."""
    question: str = Field(..., description="Original user question")
    session_id: str = Field(..., description="Session identifier")
    query_type: Optional[QueryType] = Field(
        default=None, description="Determined query type")
    retrieved_docs: List[str] = Field(
        default=[], description="Retrieved documents")
    web_results: List[str] = Field(
        default=[], description="Web search results")
    final_answer: Optional[str] = Field(
        default=None, description="Final generated answer")
    confidence: float = Field(default=0.0, description="Confidence score")
    sources: List[str] = Field(default=[], description="Information sources")
    chat_history: List[ChatMessage] = Field(
        default=[], description="Conversation history")
    error: Optional[str] = Field(
        default=None, description="Error message if any")


class DocumentChunk(BaseModel):
    """Model for document chunks in vector store."""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(...,
                                     description="Metadata including source, page, etc.")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding")


class RouterDecision(BaseModel):
    """Model for router agent decisions."""
    query_type: QueryType = Field(..., description="Determined query type")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Decision confidence")
    reasoning: str = Field(..., description="Reasoning behind the decision")
    requires_clarification: bool = Field(
        default=False, description="Whether query needs clarification")
