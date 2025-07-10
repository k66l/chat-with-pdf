"""LLM service for Google AI Studio integration."""

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any, Optional
import structlog

from ..core.config import settings
from ..core.models import ChatMessage

logger = structlog.get_logger(__name__)


class LLMService:
    """Service for interacting with Google AI Studio LLMs."""

    def __init__(self):
        """Initialize the LLM service."""
        self.api_key = settings.google_api_key
        self.model_name = settings.model_name
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

        # Configure Google AI
        genai.configure(api_key=self.api_key)

        # Initialize LangChain Google GenAI
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            google_api_key=self.api_key
        )

        logger.info("LLM service initialized", model=self.model_name)

    async def generate_response(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using the LLM."""
        try:
            # Convert to LangChain message format
            langchain_messages = []

            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))

            for msg in messages:
                if msg.role == "user":
                    langchain_messages.append(
                        HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    langchain_messages.append(AIMessage(content=msg.content))

            # Generate response
            response = await self.llm.ainvoke(langchain_messages)

            logger.info("Generated LLM response", length=len(response.content))
            return response.content

        except Exception as e:
            logger.error("Error generating LLM response", error=str(e))
            raise

    async def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response from a single prompt."""
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if response and hasattr(response, 'content') else ""
            
            if not content or not content.strip():
                logger.warning("LLM returned empty response for simple prompt")
                return ""
            
            return content.strip()
        except Exception as e:
            logger.error("Error generating simple response", error=str(e))
            raise

    async def classify_query_intent(self, question: str) -> Dict[str, Any]:
        """Classify the intent of a user question."""
        system_prompt = """You are a query classifier for a research paper Q&A system. The system has academic papers about Text-to-SQL research but needs web search for current events.

        Analyze the user's question and determine:
        1. Whether it should be answered using PDF documents (research papers) or web search
        2. How confident you are in this decision (0.0 to 1.0)
        3. Whether the question is ambiguous and needs clarification

        Guidelines:

        PDF SEARCH (use when question is about):
        - Text-to-SQL research, methods, techniques, benchmarks
        - Academic findings, experimental results, model performance
        - Specific research papers, authors, datasets mentioned in academic literature
        - Technical methods, algorithms, architectures described in papers
        - Prompting strategies for LLMs in research context
        - Specific table/section references with academic context (e.g., "Table 3 in Zhang et al. 2024")
        - Academic metrics and datasets (Spider, WikiSQL, execution accuracy, EX %)
        - Research comparisons and benchmarks

        WEB SEARCH (use when question is about):
        - Current events, recent releases, announcements (this week/month/year)
        - AI model releases, updates, launches ("When did X release?", "When was X launched?")
        - Company announcements, product launches, recent updates
        - "What did [company] release recently/this month/lately?"
        - "When did [model/product] release/launch?"
        - Real-time information, breaking news, latest developments
        - Current market conditions, recent trends, live data
        - Information that changes frequently or is time-sensitive
        - Questions about release dates, launch dates, availability dates

        AMBIGUOUS (use when):
        - Question contains vague terms WITHOUT specific context:
          * "enough" (without specifying for what purpose/dataset)
          * "good" or "best" (without criteria or comparison baseline)
          * "optimal" (without defining optimization target)
          * "sufficient" (without specifying requirements)
        - Missing critical details:
          * Dataset not specified when asking about performance
          * No accuracy target or metric mentioned
          * No specific use case or domain context
          * No specific type of model or prediction task mentioned
        - Multiple possible interpretations exist
        - Question is too general or lacks specificity
        - Questions about "good models" without specifying domain/task

        IMPORTANT: If a question contains SPECIFIC CONTEXT, do NOT mark as ambiguous:
        - ✅ "How many examples for good accuracy on Spider dataset?" → PDF_SEARCH (has dataset)
        - ✅ "What is good accuracy using exact match metric?" → PDF_SEARCH (has metric)
        - ✅ "How many examples for Text-to-SQL tasks?" → PDF_SEARCH (has domain)
        - ✅ "Which prompt template gave highest accuracy in Zhang et al. 2024?" → PDF_SEARCH (has author/year)
        - ✅ "What does Table 3 show about Text-to-SQL performance?" → PDF_SEARCH (has table + academic context)
        - ✅ "What is a good model for Text-to-SQL prediction?" → PDF_SEARCH (has specific domain)
        - ❌ "How many examples are enough?" → AMBIGUOUS (no context)
        - ❌ "What is good accuracy?" → AMBIGUOUS (no dataset/metric)
        - ❌ "What is a good model for prediction?" → AMBIGUOUS (no specific domain/task)

        CRITICAL: Be especially strict about vague quantitative terms ONLY when no context:
        - "How many X are enough?" → Check for dataset/domain context first
        - "What is good accuracy?" → Check for dataset/metric context first
        - "Which method is best?" → Check for task/criteria context first
        - "How to optimize X?" → Check for specific goals/constraints first

        SPECIAL CASES:
        - Questions with table/section references AND academic context → PDF_SEARCH
        - Questions with author names AND years → PDF_SEARCH
        - Questions with specific datasets (Spider, WikiSQL) → PDF_SEARCH
        - Questions with specific metrics (execution accuracy, EX %) → PDF_SEARCH

        Be especially careful with temporal keywords:
        - "this month", "recently", "latest release", "just announced" → WEB SEARCH
        - "current", "now", "today", "this week" → WEB SEARCH
        - Research methods, academic findings, paper content → PDF SEARCH

        Respond in JSON format:
        {
            "query_type": "pdf_search|web_search|ambiguous",
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "requires_clarification": true/false
        }"""

        try:
            prompt = f"{system_prompt}\n\nUser question: {question}"
            response = await self.generate_simple_response(prompt)

            # Try to parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON response, using fallback")
                return {
                    "query_type": "pdf_search",
                    "confidence": 0.5,
                    "reasoning": "Could not parse classifier response",
                    "requires_clarification": False
                }
        except Exception as e:
            logger.error("Error classifying query intent", error=str(e))
            raise

    async def synthesize_answer(
        self,
        question: str,
        context: str,
        sources: List[str],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Synthesize a final answer using context and sources."""

        # Build conversation context
        conversation_context = ""
        if chat_history:
            recent_messages = chat_history[-5:]  # Last 5 messages for context
            conversation_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])

        # Enhanced prompt with better context handling
        system_prompt = f"""WRITE EXACTLY 2 SENTENCES. NO MORE. NO BULLET POINTS. NO LISTS.

        Question: {question}

        Research Papers Context:
        {context[:8000]}{'...' if len(context) > 8000 else ''}

        CRITICAL INSTRUCTIONS:
        - EXACTLY 2 sentences maximum
        - Use ONLY information explicitly written in the context above
        
        FOR EXPLANATION QUESTIONS - MANDATORY FORMAT:
        - FIRST SENTENCE: Start with "[Concept] is..." or "[Concept] involves..." and explain WHAT it is conceptually
        - SECOND SENTENCE: Describe the paper's specific findings or contributions about this concept
        - ABSOLUTELY FORBIDDEN: Starting with "In [concept]..." or "Demonstrations are..." or experimental details
        - REQUIRED: Begin with the concept name followed by definition of what it actually means
        
        GOOD EXAMPLE: "Cross-domain Text-to-SQL is a challenging task where models generate SQL queries for databases with different schemas than those seen in training. Chang and Fosler-Lussier (2023) found that..."
        
        BAD EXAMPLE: "In Cross-domain Text-to-SQL, demonstrations are..." or "Demonstrations consist of..."
        
        - Extract conceptual understanding first, then paper contributions
        - Focus on helping readers understand the concept itself, not just experimental setup
        - Do NOT include inline citations like (Document X, Page Y)

        BEFORE ANSWERING: Carefully scan the entire context for relevant information including:
        - Conceptual definitions and explanations of the topic
        - Key contributions and findings from the mentioned authors/papers
        - Methodological approaches and their significance
        - Specific numerical results and percentages that support the explanation
        - Performance metrics and benchmarks when relevant
        - Experimental findings and conclusions
        - Comparative analysis or advantages of the approach

        REMEMBER: If this is an explanation question, your answer MUST start with the concept name followed by "is" or "involves" and then define what it means. Do NOT start with experimental details.
        
        Answer in EXACTLY 2 sentences with all relevant information from the context:"""

        try:
            logger.info("Synthesizing answer",
                        question_length=len(question),
                        context_length=len(context),
                        sources_count=len(sources))

            response = await self.generate_simple_response(system_prompt)

            # Check if response is empty or None
            if not response or not response.strip():
                logger.warning(
                    "Empty response from LLM, generating fallback from context")

                # Create a basic answer from the first part of the context
                context_preview = context[:1000] + \
                    "..." if len(context) > 1000 else context
                fallback_answer = f"""Based on the research papers, here's what I found about {question}:

                {context_preview}

                For more specific information, please try asking about particular aspects like methods, results, or datasets."""

                return fallback_answer

            logger.info("Successfully synthesized answer",
                        response_length=len(response))
            return response.strip()

        except Exception as e:
            logger.error("Error synthesizing answer", error=str(e))
            # Provide a more helpful fallback response with actual content
            context_preview = context[:800] + \
                "..." if len(context) > 800 else context
            return f"""I found relevant information about {question} in the research papers:

            {context_preview}

            There was a technical issue generating a complete response, but the above content should help answer your question."""


# Global LLM service instance
llm_service = LLMService()
