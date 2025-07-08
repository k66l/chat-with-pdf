"""PDF Agent for RAG-based document retrieval and question answering."""

from typing import List, Dict, Any, Tuple
import structlog

from ..core.models import ChatMessage
from ..services.vector_store import vector_store_service
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class PDFAgent:
    """Agent for answering questions using PDF documents via RAG."""

    def __init__(self):
        """Initialize the PDF Agent."""
        self.max_retrieved_docs = 5
        self.score_threshold = 0.3  # Lowered from 0.5 to be more permissive
        logger.info("PDF Agent initialized")

    async def search_documents(
        self,
        question: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents in the vector store."""
        try:
            if top_k is None:
                top_k = self.max_retrieved_docs
            if score_threshold is None:
                score_threshold = self.score_threshold

            logger.info(
                "Searching documents",
                question_length=len(question),
                top_k=top_k,
                score_threshold=score_threshold
            )

            # Perform vector search
            results = await vector_store_service.search(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold
            )

            logger.info(
                "Document search completed",
                results_count=len(results)
            )

            return results

        except Exception as e:
            logger.error("Error searching documents", error=str(e))
            return []

    async def answer_question(
        self,
        question: str,
        chat_history: List[ChatMessage] = None
    ) -> Tuple[str, List[str], float]:
        """Answer a question using retrieved documents."""
        try:
            logger.info("Answering question using PDF documents")

            # Check if this is clearly an out-of-scope query before searching
            if await self._is_out_of_scope_query(question):
                out_of_scope_response = (
                    "This question appears to be about current events or recent developments "
                    "that wouldn't be covered in academic research papers. You might want to "
                    "try asking about specific research methods, findings, or technical approaches "
                    "related to Text-to-SQL instead."
                )
                return out_of_scope_response, [], 0.1

            # Search for relevant documents
            retrieved_docs = await self.search_documents(question)

            if not retrieved_docs:
                no_docs_response = (
                    "I couldn't find any relevant information in the available PDF documents "
                    "to answer your question. The documents might not contain information "
                    "about this topic, or you might want to try rephrasing your question."
                )
                return no_docs_response, [], 0.0

            # Prepare context from retrieved documents
            context_parts = []
            sources = []

            for i, doc in enumerate(retrieved_docs):
                content = doc['content']
                metadata = doc['metadata']
                score = doc['score']

                # Add document context
                source_info = f"{metadata.get('source', 'Unknown')} (Page {metadata.get('page', 'N/A')})"
                context_parts.append(
                    f"Document {i+1} ({source_info}):\n{content}")

                # Track unique sources
                if source_info not in sources:
                    sources.append(source_info)

            # Combine context
            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            answer = await llm_service.synthesize_answer(
                question=question,
                context=context,
                sources=sources,
                chat_history=chat_history
            )

            # Calculate confidence based on retrieval scores
            avg_score = sum(doc['score']
                            for doc in retrieved_docs) / len(retrieved_docs)
            confidence = min(avg_score, 1.0)

            logger.info(
                "Generated answer from PDF documents",
                sources_count=len(sources),
                confidence=confidence
            )

            return answer, sources, confidence

        except Exception as e:
            logger.error(
                "Error answering question with PDF documents", error=str(e))
            error_response = (
                "I encountered an error while searching the PDF documents. "
                "Please try again or rephrase your question."
            )
            return error_response, [], 0.0

    async def _is_out_of_scope_query(self, question: str) -> bool:
        """Check if a query is clearly out of scope for academic papers."""
        try:
            # Simple keyword-based detection for clearly temporal/current events
            temporal_keywords = [
                "this month", "this week", "recently", "latest release",
                "just announced", "breaking", "today", "yesterday",
                "current", "now", "this year 2024", "what did", "released"
            ]

            company_keywords = [
                "openai", "google", "microsoft", "meta", "apple",
                "amazon", "tesla", "nvidia"
            ]

            question_lower = question.lower()

            # Check if question contains temporal + company keywords
            has_temporal = any(
                keyword in question_lower for keyword in temporal_keywords)
            has_company = any(
                keyword in question_lower for keyword in company_keywords)

            if has_temporal and has_company:
                logger.info("Detected out-of-scope query", question=question)
                return True

            # Check for common current events patterns
            current_events_patterns = [
                "what did .* release",
                "latest .* announcement",
                "recent .* news",
                ".* this month",
                ".* this week"
            ]

            import re
            for pattern in current_events_patterns:
                if re.search(pattern, question_lower):
                    logger.info("Detected current events pattern",
                                question=question, pattern=pattern)
                    return True

            return False

        except Exception as e:
            logger.error("Error checking query scope", error=str(e))
            return False

    async def get_document_context(
        self,
        question: str,
        max_context_length: int = 4000
    ) -> Tuple[str, List[str]]:
        """Get relevant document context for a question."""
        try:
            # Search for relevant documents
            retrieved_docs = await self.search_documents(question)

            if not retrieved_docs:
                return "", []

            # Build context within length limit
            context_parts = []
            sources = []
            current_length = 0

            for doc in retrieved_docs:
                content = doc['content']
                metadata = doc['metadata']

                # Check if adding this document would exceed length limit
                if current_length + len(content) > max_context_length:
                    # Try to fit part of the document
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if substantial space remains
                        content = content[:remaining_space] + "..."
                        context_parts.append(content)
                        current_length += len(content)
                    break

                context_parts.append(content)
                current_length += len(content)

                # Track sources
                source_info = f"{metadata.get('source', 'Unknown')} (Page {metadata.get('page', 'N/A')})"
                if source_info not in sources:
                    sources.append(source_info)

            context = "\n\n".join(context_parts)

            logger.info(
                "Retrieved document context",
                context_length=len(context),
                sources_count=len(sources)
            )

            return context, sources

        except Exception as e:
            logger.error("Error getting document context", error=str(e))
            return "", []

    def get_available_documents(self) -> List[Dict[str, Any]]:
        """Get information about available documents in the vector store."""
        try:
            # Get unique sources and their statistics
            sources_info = {}

            for doc in vector_store_service.documents:
                metadata = doc['metadata']
                source = metadata.get('source', 'Unknown')

                if source not in sources_info:
                    sources_info[source] = {
                        'file_name': source,
                        'total_chunks': 0,
                        'pages': set()
                    }

                sources_info[source]['total_chunks'] += 1
                if 'page' in metadata:
                    sources_info[source]['pages'].add(metadata['page'])

            # Convert to list format
            documents_list = []
            for source, info in sources_info.items():
                documents_list.append({
                    'file_name': info['file_name'],
                    'total_chunks': info['total_chunks'],
                    'total_pages': len(info['pages']),
                    'pages': sorted(list(info['pages']))
                })

            return sorted(documents_list, key=lambda x: x['file_name'])

        except Exception as e:
            logger.error("Error getting available documents", error=str(e))
            return []


# Global PDF agent instance
pdf_agent = PDFAgent()
