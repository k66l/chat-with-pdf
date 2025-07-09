"""LangGraph orchestrator for coordinating multi-agent workflow."""

from typing import Dict, Any, List
import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.models import AgentState, QueryType, OrchestratorResponse, ChatMessage
from .router_agent import router_agent
from .pdf_agent import pdf_agent
from .web_search_agent import web_search_agent
from .memory_agent import memory_agent
from .evaluation_agent import evaluation_agent

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """LangGraph-based orchestrator for multi-agent workflow."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.graph = self._build_graph()
        # Configurable thresholds and settings
        self.fallback_confidence_threshold = self._get_fallback_confidence_threshold()
        self.ambiguous_query_confidence = self._get_ambiguous_query_confidence()
        self.error_confidence = self._get_error_confidence()
        self.no_results_indicators = self._get_no_results_indicators()
        logger.info("Agent Orchestrator initialized")

    def _get_fallback_confidence_threshold(self) -> float:
        """Get configurable fallback confidence threshold."""
        from ..core.config import settings
        return getattr(settings, 'fallback_confidence_threshold', 0.1)

    def _get_ambiguous_query_confidence(self) -> float:
        """Get configurable ambiguous query confidence."""
        from ..core.config import settings
        return getattr(settings, 'ambiguous_query_confidence', 0.3)

    def _get_error_confidence(self) -> float:
        """Get configurable error confidence."""
        from ..core.config import settings
        return getattr(settings, 'error_confidence', 0.0)

    def _get_no_results_indicators(self) -> List[str]:
        """Get configurable no results indicators."""
        from ..core.config import settings
        return getattr(settings, 'no_results_indicators', [
            "couldn't find any relevant information",
            "appears to be about current events",
            "recent developments that wouldn't be covered",
            "documents might not contain information",
            "do not contain information about",
            "don't have information about",
            "not covered in the",
            "no information about",
            "not found in the papers",
            "papers do not contain"
        ])

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        try:
            # Create the state graph
            workflow = StateGraph(AgentState)

            # Add nodes (agents)
            workflow.add_node("route_query", self._route_query_node)
            workflow.add_node("handle_pdf_search",
                              self._handle_pdf_search_node)
            workflow.add_node("handle_web_search",
                              self._handle_web_search_node)
            workflow.add_node("handle_ambiguous", self._handle_ambiguous_node)
            workflow.add_node("finalize_response",
                              self._finalize_response_node)

            # Set entry point
            workflow.set_entry_point("route_query")

            # Add conditional edges
            workflow.add_conditional_edges(
                "route_query",
                self._routing_condition,
                {
                    "pdf_search": "handle_pdf_search",
                    "web_search": "handle_web_search",
                    "ambiguous": "handle_ambiguous"
                }
            )

            # Add edges to finalization
            workflow.add_edge("handle_pdf_search", "finalize_response")
            workflow.add_edge("handle_web_search", "finalize_response")
            workflow.add_edge("handle_ambiguous", "finalize_response")

            # End the workflow
            workflow.add_edge("finalize_response", END)

            # Compile the graph
            memory = MemorySaver()
            compiled_graph = workflow.compile(checkpointer=memory)

            logger.info("LangGraph workflow compiled successfully")
            return compiled_graph

        except Exception as e:
            logger.error("Error building LangGraph workflow", error=str(e))
            raise

    async def _route_query_node(self, state: AgentState) -> AgentState:
        """Route the query using the router agent."""
        try:
            logger.info("Routing query", session_id=state.session_id)

            # Get routing decision
            decision = await router_agent.route_query(state.question)

            # Update state
            state.query_type = decision.query_type
            state.confidence = decision.confidence

            logger.info(
                "Query routed",
                query_type=decision.query_type.value,
                confidence=decision.confidence
            )

            return state

        except Exception as e:
            logger.error("Error in route_query_node", error=str(e))
            state.error = f"Routing error: {str(e)}"
            return state

    async def _handle_pdf_search_node(self, state: AgentState) -> AgentState:
        """Handle PDF search using the PDF agent."""
        try:
            logger.info("Handling PDF search", session_id=state.session_id)

            # Get chat history
            chat_history = memory_agent.get_chat_history(state.session_id)

            # Perform PDF search
            answer, sources, confidence = await pdf_agent.answer_question(
                question=state.question,
                chat_history=chat_history
            )

            # Check if PDF search found no relevant results or detected out-of-scope query
            should_fallback = (
                confidence <= self.fallback_confidence_threshold or  # Very low confidence
                not sources or  # No sources found
                any(indicator in answer.lower()
                    for indicator in self.no_results_indicators)
            )

            if should_fallback:
                logger.info("PDF search found no relevant results, falling back to web search",
                            confidence=confidence, sources_count=len(sources))

                # Try web search instead
                web_answer, web_sources, web_confidence = await web_search_agent.answer_question(
                    question=state.question,
                    chat_history=chat_history
                )

                # Update state with web search results
                state.final_answer = web_answer
                state.sources = web_sources
                state.confidence = web_confidence
                # Update query type to reflect actual search used
                state.query_type = QueryType.WEB_SEARCH

                logger.info("Fallback to web search completed",
                            sources_count=len(web_sources), web_confidence=web_confidence)
            else:
                # Use PDF search results
                state.final_answer = answer
                state.sources = sources
                state.confidence = confidence

                logger.info("PDF search completed", sources_count=len(sources))

            return state

        except Exception as e:
            logger.error("Error in handle_pdf_search_node", error=str(e))
            state.error = f"PDF search error: {str(e)}"
            return state

    async def _handle_web_search_node(self, state: AgentState) -> AgentState:
        """Handle web search using the web search agent."""
        try:
            logger.info("Handling web search", session_id=state.session_id)

            # Get chat history
            chat_history = memory_agent.get_chat_history(state.session_id)

            # Perform web search
            answer, sources, confidence = await web_search_agent.answer_question(
                question=state.question,
                chat_history=chat_history
            )

            # Update state
            state.final_answer = answer
            state.sources = sources
            state.confidence = confidence

            logger.info("Web search completed", sources_count=len(sources))
            return state

        except Exception as e:
            logger.error("Error in handle_web_search_node", error=str(e))
            state.error = f"Web search error: {str(e)}"
            return state

    async def _handle_ambiguous_node(self, state: AgentState) -> AgentState:
        """Handle ambiguous queries."""
        try:
            logger.info("Handling ambiguous query",
                        session_id=state.session_id)

            # Generate clarification
            clarification = await router_agent.handle_ambiguous_query(state.question)

            # Update state
            state.final_answer = clarification
            state.sources = []
            # Configurable confidence for ambiguous queries
            state.confidence = self.ambiguous_query_confidence

            logger.info("Ambiguous query handled")
            return state

        except Exception as e:
            logger.error("Error in handle_ambiguous_node", error=str(e))
            state.error = f"Ambiguous query handling error: {str(e)}"
            return state

    async def _finalize_response_node(self, state: AgentState) -> AgentState:
        """Finalize the response and update memory."""
        try:
            logger.info("Finalizing response", session_id=state.session_id)

            # Handle errors
            if state.error:
                state.final_answer = f"I encountered an error: {state.error}. Please try again."
                state.confidence = self.error_confidence

            # Evaluate the answer quality using the Evaluation Agent
            if state.final_answer and state.query_type:
                chat_history = memory_agent.get_chat_history(state.session_id)

                evaluation_result = await evaluation_agent.evaluate_answer(
                    question=state.question,
                    answer=state.final_answer,
                    sources=state.sources,
                    query_type=state.query_type,
                    retrieval_confidence=state.confidence,
                    chat_history=chat_history
                )

                # Update confidence with evaluation result
                state.confidence = evaluation_result['overall_confidence']

                logger.info(
                    "Answer evaluated",
                    overall_confidence=evaluation_result['overall_confidence'],
                    quality_rating=evaluation_result['quality_rating']
                )

            # Add messages to memory
            memory_agent.add_user_message(
                session_id=state.session_id,
                content=state.question,
                metadata={
                    "query_type": state.query_type.value if state.query_type else "unknown"}
            )

            memory_agent.add_assistant_message(
                session_id=state.session_id,
                content=state.final_answer or "No response generated",
                metadata={
                    "sources": state.sources,
                    "confidence": state.confidence,
                    "query_type": state.query_type.value if state.query_type else "unknown"
                }
            )

            logger.info("Response finalized and memory updated")
            return state

        except Exception as e:
            logger.error("Error in finalize_response_node", error=str(e))
            state.error = f"Finalization error: {str(e)}"
            return state

    def _routing_condition(self, state: AgentState) -> str:
        """Determine which path to take based on query type."""
        if state.query_type == QueryType.PDF_SEARCH:
            return "pdf_search"
        elif state.query_type == QueryType.WEB_SEARCH:
            return "web_search"
        elif state.query_type == QueryType.AMBIGUOUS:
            return "ambiguous"
        else:
            # Default to PDF search
            return "pdf_search"

    async def process_question(self, question: str, session_id: str) -> OrchestratorResponse:
        """Process a question through the multi-agent workflow."""
        try:
            logger.info("Processing question", session_id=session_id)

            # Create initial state
            initial_state = AgentState(
                question=question,
                session_id=session_id
            )

            # Run the workflow
            config = {"configurable": {"thread_id": session_id}}
            result = await self.graph.ainvoke(initial_state, config)

            # Debug logging
            logger.info("LangGraph workflow completed",
                        result_type=type(result).__name__,
                        result_keys=list(result.keys()) if isinstance(result, dict) else "not_dict")

            # Convert result back to AgentState if it's a dict
            if isinstance(result, dict):
                try:
                    # Create AgentState from dict
                    final_state = AgentState(**result)
                except Exception as state_error:
                    logger.error("Error converting dict to AgentState",
                                 error=str(state_error),
                                 result_keys=list(result.keys()))
                    # Fallback: access dict directly
                    final_state = type('obj', (object,), {
                        'final_answer': result.get('final_answer'),
                        'sources': result.get('sources', []),
                        'query_type': result.get('query_type', QueryType.PDF_SEARCH),
                        'confidence': result.get('confidence', 0.0)
                    })()
            else:
                final_state = result

            # Create response using the final state
            response = OrchestratorResponse(
                answer=final_state.final_answer or "No answer generated",
                sources=final_state.sources or [],
                query_type=final_state.query_type or QueryType.PDF_SEARCH,
                confidence=final_state.confidence or 0.0
            )

            logger.info(
                "Question processed successfully",
                session_id=session_id,
                query_type=response.query_type.value
            )

            return response

        except Exception as e:
            logger.error("Error processing question",
                         session_id=session_id, error=str(e))

            # Return error response
            return OrchestratorResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try again.",
                sources=[],
                query_type=QueryType.PDF_SEARCH,
                confidence=0.0
            )


# Global orchestrator instance
orchestrator = AgentOrchestrator()
