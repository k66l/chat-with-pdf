"""Web Search Agent using Tavily API for external search."""

from typing import List, Dict, Any, Tuple
import structlog
from tavily import TavilyClient

from ..core.config import settings
from ..core.models import ChatMessage
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class WebSearchAgent:
    """Agent for answering questions using web search."""

    def __init__(self):
        """Initialize the Web Search Agent."""
        self.tavily_api_key = settings.tavily_api_key
        self.client = TavilyClient(api_key=self.tavily_api_key)
        self.max_results = self._get_max_results()
        self.context_length_limit = self._get_context_length_limit()
        self.fallback_preview_length = self._get_fallback_preview_length()
        self.enhancement_message_limit = self._get_enhancement_message_limit()
        self.current_info_results = self._get_current_info_results()
        logger.info("Web Search Agent initialized")

    def _get_max_results(self) -> int:
        """Get configurable maximum search results."""
        return getattr(settings, 'web_search_max_results', 5)

    def _get_context_length_limit(self) -> int:
        """Get configurable context length limit."""
        return getattr(settings, 'web_search_context_limit', 3000)

    def _get_fallback_preview_length(self) -> int:
        """Get configurable fallback preview length."""
        return getattr(settings, 'web_search_fallback_preview', 1000)

    def _get_enhancement_message_limit(self) -> int:
        """Get configurable enhancement message limit."""
        return getattr(settings, 'web_search_enhancement_messages', 3)

    def _get_current_info_results(self) -> int:
        """Get configurable current info search results."""
        return getattr(settings, 'web_search_current_info_results', 3)

    async def search_web(
        self,
        query: str,
        max_results: int = None
    ) -> List[Dict[str, Any]]:
        """Perform web search using Tavily."""
        try:
            if max_results is None:
                max_results = self.max_results

            logger.info(
                "Performing web search",
                query_length=len(query),
                max_results=max_results
            )

            # Use Tavily search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_images=False
            )

            # Process results
            results = []
            if 'results' in response:
                for result in response['results']:
                    results.append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': result.get('url', ''),
                        'score': result.get('score', 0.0)
                    })

            logger.info(
                "Web search completed",
                results_count=len(results)
            )

            return results

        except Exception as e:
            logger.error("Error performing web search", error=str(e))
            return []

    async def answer_question(
        self,
        question: str,
        chat_history: List[ChatMessage] = None
    ) -> Tuple[str, List[str], float]:
        """Answer a question using web search results."""
        try:
            logger.info("Answering question using web search")

            # Enhance query for better search results
            enhanced_query = await self._enhance_search_query(question, chat_history)

            # Perform web search
            search_results = await self.search_web(enhanced_query)

            if not search_results:
                no_results_response = (
                    "I couldn't find any relevant information on the web "
                    "to answer your question. You might want to try rephrasing "
                    "your question or being more specific."
                )
                return no_results_response, [], 0.0

            # Prepare context from search results
            context_parts = []
            sources = []

            for i, result in enumerate(search_results):
                title = result['title']
                content = result['content']
                url = result['url']

                # Add search result context
                context_parts.append(f"Source {i+1} - {title}:\n{content}")
                sources.append(f"{title} ({url})")

            # Combine context
            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            web_prompt = f"""Answer the user's question based on current web search results.

            Question: {question}

            Search Results:
            {context[:self.context_length_limit]}{'...' if len(context) > self.context_length_limit else ''}

            Instructions:
            - Provide current, accurate information
            - Cite sources when possible
            - Be concise and informative

            Answer:"""

            answer = await llm_service.generate_simple_response(web_prompt)

            # Calculate confidence based on search result scores
            if search_results:
                avg_score = sum(result.get('score', 0.5)
                                for result in search_results) / len(search_results)
                confidence = min(avg_score, 1.0)
            else:
                confidence = 0.0

            # Check if response is empty or None
            if not answer or not answer.strip():
                logger.warning(
                    "Empty response from LLM for web search, generating fallback")

                # Create a basic answer from the search results
                context_preview = (context[:self.fallback_preview_length] +
                                   "..." if len(context) > self.fallback_preview_length else context)
                fallback_answer = f"""Based on current web search results about {question}:

                {context_preview}

                Sources: {', '.join(sources)}

                For the most up-to-date information, please check the sources above."""

                return fallback_answer, sources, confidence

            logger.info("Successfully generated web search answer",
                        response_length=len(answer))

            return answer, sources, confidence

        except Exception as e:
            logger.error(
                "Error answering question with web search", error=str(e))
            error_response = (
                "I encountered an error while searching the web. "
                "Please try again or check your internet connection."
            )
            return error_response, [], 0.0

    async def _enhance_search_query(
        self,
        question: str,
        chat_history: List[ChatMessage] = None
    ) -> str:
        """Enhance the search query for better results."""
        try:
            # Build conversation context if available
            conversation_context = ""
            if chat_history:
                # Use configurable number of recent messages for context
                recent_messages = chat_history[-self.enhancement_message_limit:]
                conversation_context = "\n".join([
                    f"{msg.role}: {msg.content}" for msg in recent_messages
                ])

            enhance_prompt = f"""Transform the following user question into an effective web search query.
            Make it more specific and search-friendly while preserving the original intent.

            User question: {question}

            {f'Conversation context:{conversation_context}' if conversation_context else ''}

            Guidelines:
            - Add relevant keywords
            - Remove ambiguous words
            - Make it specific for current/recent information
            - Keep it concise but comprehensive

            Enhanced search query:"""

            enhanced_query = await llm_service.generate_simple_response(enhance_prompt)

            # Clean up the response (remove quotes, extra text)
            enhanced_query = enhanced_query.strip().strip('"\'')

            # Fallback to original question if enhancement fails
            if not enhanced_query or len(enhanced_query) < 3:
                enhanced_query = question

            logger.info(
                "Enhanced search query",
                original=question,
                enhanced=enhanced_query
            )

            return enhanced_query

        except Exception as e:
            logger.error("Error enhancing search query", error=str(e))
            return question  # Fallback to original question

    async def get_current_info(self, topic: str) -> Dict[str, Any]:
        """Get current information about a specific topic."""
        try:
            # Search for recent information
            query = f"latest {topic} 2024 recent developments"
            results = await self.search_web(query, max_results=self.current_info_results)

            if not results:
                return {
                    'success': False,
                    'message': f'No current information found about {topic}',
                    'results': []
                }

            return {
                'success': True,
                'message': f'Found current information about {topic}',
                'results': results,
                'query_used': query
            }

        except Exception as e:
            logger.error("Error getting current info",
                         topic=topic, error=str(e))
            return {
                'success': False,
                'message': f'Error retrieving current information: {str(e)}',
                'results': []
            }


# Global web search agent instance
web_search_agent = WebSearchAgent()
