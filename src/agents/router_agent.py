"""Router Agent for deciding between PDF search and web search."""

from typing import Dict, Any
import structlog

from ..core.models import RouterDecision, QueryType
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class RouterAgent:
    """Agent that routes queries to appropriate handlers."""

    def __init__(self):
        """Initialize the Router Agent."""
        self.confidence_threshold = 0.3
        logger.info("Router Agent initialized")

    async def route_query(self, question: str) -> RouterDecision:
        """Analyze the query and decide on routing."""
        try:
            logger.info("Routing query", question_length=len(question))

            # First check for obvious web search patterns
            if self._is_obvious_web_search(question):
                return RouterDecision(
                    query_type=QueryType.WEB_SEARCH,
                    confidence=0.9,
                    reasoning="Question contains clear temporal/current events indicators",
                    requires_clarification=False
                )

            # THEN check for obviously ambiguous patterns (moved up in priority)
            if self._is_obviously_ambiguous(question):
                return RouterDecision(
                    query_type=QueryType.AMBIGUOUS,
                    confidence=0.8,
                    reasoning="Question contains vague terms that require clarification",
                    requires_clarification=True
                )

            # Use LLM to classify the query only if no obvious patterns detected
            classification = await llm_service.classify_query_intent(question)

            # Map the classification to our models
            query_type_map = {
                "pdf_search": QueryType.PDF_SEARCH,
                "web_search": QueryType.WEB_SEARCH,
                "ambiguous": QueryType.AMBIGUOUS
            }

            query_type = query_type_map.get(
                classification.get("query_type", "pdf_search"),
                QueryType.PDF_SEARCH
            )

            confidence = float(classification.get("confidence", 0.5))
            reasoning = classification.get(
                "reasoning", "Classification completed")
            requires_clarification = classification.get(
                "requires_clarification", False)

            # Apply confidence threshold logic
            if confidence < self.confidence_threshold:
                # Low confidence, mark as ambiguous
                query_type = QueryType.AMBIGUOUS
                requires_clarification = True
                reasoning += f" (Low confidence: {confidence})"

            decision = RouterDecision(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                requires_clarification=requires_clarification
            )

            logger.info(
                "Query routing decision made",
                query_type=decision.query_type.value,
                confidence=decision.confidence,
                requires_clarification=decision.requires_clarification
            )

            return decision

        except Exception as e:
            logger.error("Error in query routing", error=str(e))
            # Fallback to PDF search on error
            return RouterDecision(
                query_type=QueryType.PDF_SEARCH,
                confidence=0.5,
                reasoning=f"Fallback due to routing error: {str(e)}",
                requires_clarification=False
            )

    def _is_obvious_web_search(self, question: str) -> bool:
        """Check for obvious web search patterns using simple rules."""
        try:
            question_lower = question.lower()

            # Temporal indicators for current events (expanded list)
            temporal_indicators = [
                "this month", "this week", "this year", "recently", "recent",
                "latest", "just announced", "breaking", "today", "new",
                "yesterday", "current", "now", "what did", "released",
                "announcement", "announce", "updates", "update"
            ]

            # Company/organization names (expanded list)
            companies = [
                "openai", "google", "microsoft", "meta", "apple",
                "amazon", "tesla", "nvidia", "anthropic", "deepmind",
                "facebook", "twitter", "x.com", "uber", "netflix"
            ]

            # AI model/product names that indicate current events
            ai_models = [
                "claude", "sonnet", "gpt", "chatgpt", "dall-e", "midjourney",
                "bard", "palm", "llama", "gemini", "copilot", "codex"
            ]

            # Current events patterns (more comprehensive)
            current_patterns = [
                r"(what|when) did .* (release|announce|launch)",
                r"when (was|did) .* (release|announce|launch)",
                r"(recent|latest) .* (announcement|news|release)",
                r".* (this month|this week|today|recently)",
                r"(new|latest) .* from",
                r"just (released|announced|launched)",
                r"breaking news",
                r"what.*(recently|latest).*(announce|release)",
                r"recent.*(development|announcement|news)",
                r"when.*release",
                r"release date.*",
                r".*release date"
            ]

            # Check for temporal + company combination
            has_temporal = any(
                indicator in question_lower for indicator in temporal_indicators)
            has_company = any(
                company in question_lower for company in companies)
            has_ai_model = any(
                model in question_lower for model in ai_models)

            if has_temporal and (has_company or has_ai_model):
                logger.info("Detected obvious web search query (temporal + company/model)",
                            question=question, temporal_found=True, company_found=has_company, model_found=has_ai_model)
                return True

            # Check for current events patterns
            import re
            for pattern in current_patterns:
                if re.search(pattern, question_lower):
                    logger.info("Detected obvious web search query (pattern match)",
                                question=question, pattern=pattern)
                    return True

            # Additional check for questions about companies/models without temporal words
            # but with action words that suggest current events
            if has_company or has_ai_model:
                action_words = ["announce", "release",
                                "launch", "reveal", "unveil", "introduce"]
                if any(action in question_lower for action in action_words):
                    logger.info("Detected obvious web search query (company/model + action)",
                                question=question)
                    return True

            return False

        except Exception as e:
            logger.error("Error in obvious web search detection", error=str(e))
            return False

    def _is_obviously_ambiguous(self, question: str) -> bool:
        """Check for obviously ambiguous patterns using simple rules."""
        try:
            question_lower = question.lower()

            # Vague quantitative terms
            vague_quantifiers = [
                "enough", "sufficient", "good", "best", "optimal",
                "better", "worse", "many", "few", "much", "little"
            ]

            # Context-dependent terms
            subjective_terms = [
                "good accuracy", "high performance", "low error",
                "fast enough", "slow", "efficient", "effective"
            ]

            # Ambiguous question patterns - but only if no context is provided
            ambiguous_patterns = [
                r"how many .* (enough|sufficient)$",  # Only at end of sentence
                r"^what is (good|best|optimal)$",     # Only at start and end
                r"^which .* (better|best)$",          # Only at start and end
                r"how (much|many) .* should$",        # Only at end
                # Only at end (no specific object)
                r"how to (optimize|improve)$"
            ]

            # Strong context indicators - if present, likely NOT ambiguous
            context_indicators = [
                # Specific datasets
                "spider", "wikisql", "cosql", "sparc", "bird", "dataset",
                # Specific metrics (made more specific)
                "exact match", "execution accuracy", "f1 score", "bleu score", "rouge score",
                "precision", "recall",
                # Specific models/methods
                "gpt", "codex", "t5", "bert", "chatgpt", "llama", "model",
                # Specific context prepositions with objects
                "for text-to-sql", "on spider", "using exact match", "with few-shot",
                "in zero-shot", "for cross-domain", "on wikisql", "using prompting",
                "on the spider", "using f1", "with exact match", "on dataset"
            ]

            # If question has strong context indicators, it's likely NOT ambiguous
            if any(context in question_lower for context in context_indicators):
                logger.info("Question has context indicators, not marking as ambiguous",
                            question=question)
                return False

            # Check for vague quantifiers without any specific context
            for vague_term in vague_quantifiers:
                if vague_term in question_lower:
                    # Additional check: if question is very short and vague
                    if len(question.split()) <= 8:  # Increased from 6 to catch longer vague questions
                        logger.info("Detected ambiguous query (short vague question)",
                                    question=question, vague_term=vague_term)
                        return True

            # Special check for the "enough for good accuracy" pattern specifically
            if "enough" in question_lower and "good accuracy" in question_lower:
                # This is clearly ambiguous without specific dataset/metric
                logger.info("Detected ambiguous query (enough for good accuracy pattern)",
                            question=question)
                return True

            # Check for standalone subjective terms (not in context)
            for subjective in subjective_terms:
                if subjective in question_lower and len(question.split()) <= 8:
                    logger.info("Detected ambiguous query (standalone subjective term)",
                                question=question, subjective_term=subjective)
                    return True

            # Check for ambiguous patterns only if very specific patterns
            import re
            for pattern in ambiguous_patterns:
                if re.search(pattern, question_lower):
                    logger.info("Detected ambiguous query (strict pattern match)",
                                question=question, pattern=pattern)
                    return True

            return False

        except Exception as e:
            logger.error("Error in ambiguous query detection", error=str(e))
            return False

    async def handle_ambiguous_query(self, question: str) -> str:
        """Generate clarification for ambiguous queries."""
        try:
            # Provide specific clarification based on the type of ambiguity
            question_lower = question.lower()

            if "enough" in question_lower and "examples" in question_lower:
                return """I need more specific information to answer your question about training examples:

                1. **Which dataset** are you referring to? (e.g., Spider, WikiSQL, CoSQL, BIRD)
                2. **What accuracy target** do you consider "good"? (e.g., 70%, 80%, 90% exact match)
                3. **What type of examples** are you asking about? (few-shot prompting examples, training examples, demonstration examples)
                4. **What specific metric** should I use to measure "good accuracy"? (exact match, execution accuracy, F1 score)

                Please provide these details so I can give you a precise answer based on the research papers."""

            elif "good" in question_lower and "accuracy" in question_lower:
                return """To help you understand what constitutes "good accuracy," I need clarification:

                1. **Which specific dataset** are you asking about? (Spider, WikiSQL, etc.)
                2. **What's your baseline** or comparison point?
                3. **What accuracy metric** are you interested in? (exact match, execution accuracy, etc.)
                4. **What's your specific use case or application?**

                Please specify these details for a more helpful answer."""

            elif "best" in question_lower or "optimal" in question_lower:
                return """Your question about "best" or "optimal" approaches needs more context:

                1. **What specific task** are you optimizing for?
                2. **What are your evaluation criteria?** (accuracy, speed, resource usage, etc.)
                3. **What constraints** do you have? (computational resources, data availability, etc.)
                4. **What's your specific use case or domain?**

                Please provide these details so I can give you targeted recommendations."""

            else:
                # General clarification prompt
                clarification_prompt = f"""The user asked: "{question}"

                This question contains vague terms that need clarification. Generate 2-3 specific follow-up questions to help clarify the user's intent. Focus on identifying:
                - Specific datasets, metrics, or baselines
                - Exact criteria for subjective terms like "good", "best", "enough"
                - Missing context about the specific use case or domain
                - Any quantitative targets or thresholds

                Format as a helpful clarification request that guides the user to be more specific."""

                clarification = await llm_service.generate_simple_response(clarification_prompt)

                if not clarification or not clarification.strip():
                    return "I need more specific information to answer your question. Could you provide more context about the specific dataset, metrics, and criteria you're interested in?"

                return clarification.strip()

            logger.info("Generated clarification for ambiguous query")

        except Exception as e:
            logger.error("Error generating clarification", error=str(e))
            return "I need more specific information to answer your question. Could you provide more context about the specific dataset, metrics, and criteria you're interested in?"


# Global router agent instance
router_agent = RouterAgent()
