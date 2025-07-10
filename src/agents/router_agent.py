"""Router Agent for deciding between PDF search and web search."""

from typing import Dict, Any
import structlog
import re

from ..core.models import RouterDecision, QueryType
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class RouterAgent:
    """Agent that routes queries to appropriate handlers."""

    def __init__(self):
        """Initialize the Router Agent."""
        self.confidence_threshold = 0.4  # Reduced from 0.6 to be less strict
        logger.info("Router Agent initialized",
                    confidence_threshold=self.confidence_threshold)

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

            # Check for table/section-specific academic queries that need special handling
            if self._is_table_section_specific_query(question):
                return RouterDecision(
                    query_type=QueryType.PDF_SEARCH,
                    confidence=0.85,
                    reasoning="Question requires specific table/section data from academic papers",
                    requires_clarification=False
                )

            # Check for academic queries with specific authors and datasets
            if self._is_academic_specific_query(question):
                return RouterDecision(
                    query_type=QueryType.PDF_SEARCH,
                    confidence=0.8,
                    reasoning="Question has specific academic context (author, dataset, metric)",
                    requires_clarification=False
                )

            # Check for obviously ambiguous patterns
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

            # Enhanced confidence threshold logic with detailed logging
            if confidence < self.confidence_threshold:
                # Low confidence, mark as ambiguous
                query_type = QueryType.AMBIGUOUS
                requires_clarification = True
                reasoning += f" (Low confidence: {confidence:.3f}, threshold: {self.confidence_threshold})"

                logger.info(
                    "Query marked as ambiguous due to low confidence",
                    original_confidence=confidence,
                    threshold=self.confidence_threshold,
                    original_reasoning=classification.get(
                        "reasoning", "Unknown")
                )

            decision = RouterDecision(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                requires_clarification=requires_clarification
            )

            # Enhanced logging with detailed classification information
            logger.info(
                "Query routing decision made",
                query_type=decision.query_type.value,
                confidence=decision.confidence,
                requires_clarification=decision.requires_clarification,
                llm_classification=classification.get("query_type"),
                llm_confidence=classification.get("confidence"),
                llm_reasoning=classification.get("reasoning"),
                question_preview=question[:100] +
                "..." if len(question) > 100 else question
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

    def _is_table_section_specific_query(self, question: str) -> bool:
        """Check if query requires specific table/section data from academic papers."""
        try:
            question_lower = question.lower()

            # Table/section patterns
            table_patterns = [
                r"table \d+",
                r"table \d+\.\d+",
                r"figure \d+",
                r"figure \d+\.\d+"
            ]

            section_patterns = [
                r"section \d+",
                r"section \d+\.\d+",
                r"chapter \d+",
                r"part \d+"
            ]

            # Academic context indicators (generic patterns)
            academic_patterns = [
                r'\b(research|study|experiment|benchmark|evaluation|dataset)\b',
                r'\b(accuracy|performance|score|metric|result)\b',
                r'\b(template|method|approach|technique|algorithm)\b',
                r'\b(zero-shot|few-shot|prompt|model)\b',
                r'\bet\s+al\.\s*\(\d{4}\)',  # Citation pattern
                r'\b\d+\.\d+%\b',  # Percentage pattern
            ]

            # Check for table/section references
            import re
            has_table_section = False
            for pattern in table_patterns + section_patterns:
                if re.search(pattern, question_lower):
                    has_table_section = True
                    break

            # Check for academic context using patterns
            has_academic_context = any(
                re.search(pattern, question_lower, re.IGNORECASE) for pattern in academic_patterns)

            # Query is table/section specific if it has table/section reference AND academic context
            if has_table_section and has_academic_context:
                logger.info(
                    "Detected table/section specific academic query",
                    question=question,
                    has_table_section=has_table_section,
                    has_academic_context=has_academic_context
                )
                return True

            return False

        except Exception as e:
            logger.error(
                "Error in table/section specific query detection", error=str(e))
            return False

    def _is_academic_specific_query(self, question: str) -> bool:
        """Check if query has specific academic context that should be handled by PDF search."""
        try:
            question_lower = question.lower()

            # Generic patterns for academic context
            author_patterns = [
                # Author (Year) pattern
                r'[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\)',
                # Author Year pattern
                r'[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\d{4}',
            ]

            # Generic academic patterns
            dataset_patterns = [r'\b[A-Z][a-z]*SQL\b',
                                r'\bdataset\b', r'\bbenchmark\b']
            model_patterns = [
                r'\b(model|LLM|GPT|transformer)\b', r'\b[A-Z]+-\d+[BM]?\b']
            metric_patterns = [r'\baccuracy\b', r'\bscore\b',
                               r'\bF1\b', r'\bBLEU\b', r'\bROUGE\b']
            task_patterns = [r'\btask\b', r'\bprompt\b',
                             r'\btemplate\b', r'\btable\b', r'\bfigure\b']

            # Check for author patterns
            import re
            has_author = any(re.search(pattern, question)
                             for pattern in author_patterns)

            # Check for academic context using patterns
            has_dataset = any(re.search(pattern, question_lower, re.IGNORECASE)
                              for pattern in dataset_patterns)
            has_model = any(re.search(pattern, question_lower, re.IGNORECASE)
                            for pattern in model_patterns)
            has_metric = any(re.search(pattern, question_lower, re.IGNORECASE)
                             for pattern in metric_patterns)
            has_task = any(re.search(pattern, question_lower, re.IGNORECASE)
                           for pattern in task_patterns)

            # Query is academic specific if it has author AND (dataset OR metric OR task OR model)
            # OR if it has model AND (dataset OR metric OR task)
            if (has_author and (has_dataset or has_metric or has_task)) or (has_model and (has_dataset or has_metric or has_task)):
                logger.info(
                    "Detected academic specific query",
                    question=question,
                    has_author=has_author,
                    has_dataset=has_dataset,
                    has_metric=has_metric,
                    has_task=has_task
                )
                return True

            return False

        except Exception as e:
            logger.error(
                "Error in academic specific query detection", error=str(e))
            return False

    def _is_obvious_web_search(self, question: str) -> bool:
        """Check for obvious web search patterns using improved rules."""
        try:
            question_lower = question.lower()

            # Temporal patterns for current events
            temporal_patterns = [
                r'\b(this\s+(month|week|year))\b', r'\brecently?\b', r'\blatest\b',
                r'\bjust\s+(announced|released)\b', r'\bbreaking\b', r'\btoday\b',
                r'\byesterday\b', r'\bcurrent\b', r'\bnow\b', r'\bupdate[sd]?\b'
            ]

            # Company/commercial patterns
            company_patterns = [
                r'\b(company|corp|corporation|inc|ltd)\b',
                r'\b(startup|business|enterprise|firm)\b'
            ]

            # Tech product patterns that might indicate current events
            tech_patterns = [
                r'\b[A-Z][a-z]*GPT\b', r'\bClaude\b', r'\bBard\b',
                r'\b(AI|ML)\s+(model|system|tool)\b'
            ]

            # Academic context patterns that should NOT trigger web search
            academic_patterns = [
                r'\b(research|study|experiment|benchmark|evaluation)\b',
                r'\b(paper|publication|journal|conference)\b',
                r'\bet\s+al\.\s*\(\d{4}\)', r'\b\d{4}\s*\)\b'
            ]

            # If question has strong academic indicators, it's likely NOT a web search
            import re
            if any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in academic_patterns):
                logger.info("Question has academic indicators, not marking as web search",
                            question=question)
                return False

            # Current events patterns (comprehensive)
            current_patterns = [
                r"(what|when)\s+did\s+.*\s+(release|announce|launch)",
                r"when\s+(was|did)\s+.*\s+(release|announce|launch)",
                r"(recent|latest)\s+.*\s+(announcement|news|release)",
                r"(new|latest)\s+.*\s+from",
                r"just\s+(released|announced|launched)",
                r"breaking\s+news",
                r"what.*?(recently|latest).*(announce|release)",
                r"recent.*?(development|announcement|news)",
                r"when.*?release",
                r"release\s+date"
            ]

            # Check for temporal patterns
            has_temporal = any(re.search(pattern, question_lower, re.IGNORECASE)
                               for pattern in temporal_patterns)
            has_company = any(re.search(pattern, question_lower, re.IGNORECASE)
                              for pattern in company_patterns)
            has_tech = any(re.search(pattern, question_lower, re.IGNORECASE)
                           for pattern in tech_patterns)

            if has_temporal and (has_company or has_tech):
                logger.info("Detected obvious web search query (temporal + company/tech)",
                            question=question, temporal_found=True, company_found=has_company, tech_found=has_tech)
                return True

            # Check for current events patterns
            for pattern in current_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    logger.info("Detected obvious web search query (pattern match)",
                                question=question, pattern=pattern)
                    return True

            # Additional check for action words that suggest current events
            action_patterns = [
                r'\b(announce|release|launch|reveal|unveil|introduce)\b']
            if (has_company or has_tech) and any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in action_patterns):
                logger.info("Detected obvious web search query (company/tech + action)",
                            question=question)
                return True

            return False

        except Exception as e:
            logger.error("Error in obvious web search detection", error=str(e))
            return False

    def _is_obviously_ambiguous(self, question: str) -> bool:
        """Check for obviously ambiguous patterns using improved rules."""
        try:
            question_lower = question.lower()

            # Core vague/subjective terms that need context to be meaningful
            vague_terms = [
                r'\b(good|bad|best|better|worse|great|excellent|poor|okay|perfect|special)\b',
                r'\b(big|small|large|little|huge|tiny|short|long|tall|high|low|close|far)\b', 
                r'\b(fast|slow|quick|rapid|sluggish|soon|late)\b',
                r'\b(much|many|few|little|enough|sufficient|too|minimum|maximum)\b',
                r'\b(right|wrong|correct|proper|appropriate|safe|smart|easy|hard)\b',
                r'\b(valuable|worthless|important|significant|real|true|actual)\b',
                r'\b(ready|prepared|finished|done|complete|strong|weak)\b',
                r'\b(fair|unfair|reasonable|unreasonable)\b',
                r'\b(worth|worthwhile|costly|expensive|cheap)\b'
            ]

            # Question patterns that are inherently context-dependent
            context_dependent_patterns = [
                r'^(what|which|how)\s+(is|are|was|were|will|should|would|could|can|do|does|did)\s+.*\s+(good|best|right|enough|much|big|fast|ready|worth|valuable|fair|safe|smart|easy)\b',
                r'^(is|are|was|were|will)\s+.*\s+(going\s+to\s+be|worth|good|bad|ready|enough|safe|okay|better)\b',
                r'^how\s+(much|many|big|fast|long|far|often|soon|high|close|strong)\s+.*\s+(is|are|should|would|too|enough)\b',
                r'^what.*(makes|means|defines|determines).*\s+(good|valuable|important|fair|right|special|worthwhile)\b',
                r'^when\s+should\s+(i|we|one|you)\s+.*\b(?!.*\b(dataset|model|training|algorithm|paper|research)\b)',
                r'^(should|would|could|can)\s+(i|we|one|you)\s+.*\b(?!.*\b(use|implement|apply|train|evaluate)\b)',
                r'^\w+.*(way|approach|method|choice|move|thing)\s+to\s+(get\s+started|finish|win|proceed|continue)\b(?!.*\b(training|model|algorithm)\b)',
                r'^how\s+do\s+(i|we|you)\s+(know|make|measure)\s+.*\s+(done|better|progress)\b(?!.*\b(accuracy|performance|metric)\b)'
            ]

            # Philosophical/subjective question patterns
            philosophical_patterns = [
                r'what.*(point|meaning|purpose|difference|goal|value|cost).*of\b',
                r'what.*makes.*\b(valuable|important|good|great|special|worthwhile)\b',
                r'what.*does.*\b(success|progress|done|better|perfect)\b.*look\s+like\b',
                r'is.*it.*(better|worth|good|bad|safe|okay).*to\b',
                r'what.*(real|true|actual)\s+(value|cost|difference)\b',
                r'^is\s+(this|now|it)\s+(better|worse|safer|okay)\s+than\b'
            ]

            # Ambiguous question patterns (very specific)
            ambiguous_patterns = [
                r'^(what|which)\s+is\s+(good|best|optimal)\s*[?.]?$',
                r'^(what|which)\s+is\s+(good|best|optimal)\s+\w+\s+for\s+\w+\s*[?.]?$',
                r'^how\s+(much|many)\s+.*\s+(enough|sufficient)\s*[?.]?$',
                r'^how\s+to\s+(optimize|improve)\s*[?.]?$',
                r'^how\s+long\s+does\s+it\s+take\s+to\s+train\s+(a\s+)?model\s*[?.]?$',
                r'^how\s+long\s+.*\s+train\s*[?.]?$'
            ]

            # Strong context patterns - if present, likely NOT ambiguous
            context_patterns = [
                r'\b(dataset|benchmark|metric|paper)\b',
                r'\b(accuracy|precision|recall|F1|BLEU|ROUGE)\b',
                r'\b(spider|wikisql|cosql|bird)\b',
                r'\bfor\s+(text-to-sql|sql|classification|nlp|spider|wikisql)\b',
                r'\bon\s+(spider|wikisql|cosql|bird)\b',
                r'\busing\s+(gpt|bert|transformer|neural)\b',
                r'\bwith\s+(transformer|neural|deep)\b'
            ]

            # Check for all types of vague/ambiguous patterns
            import re
            has_vague_terms = any(re.search(pattern, question_lower, re.IGNORECASE)
                                  for pattern in vague_terms)
            has_context_dependent = any(re.search(pattern, question_lower, re.IGNORECASE)
                                        for pattern in context_dependent_patterns)
            has_philosophical = any(re.search(pattern, question_lower, re.IGNORECASE)
                                    for pattern in philosophical_patterns)

            # Check for specific ambiguous patterns first
            for pattern in ambiguous_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    logger.info("Detected ambiguous query (specific pattern match)",
                                question=question, pattern=pattern)
                    return True

            # Check for context-dependent or philosophical questions first (highest priority)
            if has_context_dependent or has_philosophical:
                logger.info("Detected ambiguous query (context-dependent or philosophical)",
                            question=question,
                            has_context_dependent=has_context_dependent,
                            has_philosophical=has_philosophical)
                return True

            # If question has strong academic context indicators, it's likely NOT ambiguous
            has_context = any(re.search(pattern, question_lower, re.IGNORECASE)
                              for pattern in context_patterns)

            if has_context and not (has_context_dependent or has_philosophical):
                logger.info("Question has academic context indicators, not marking as ambiguous",
                            question=question)
                return False

            # Mark as ambiguous if contains vague terms without strong academic context
            if has_vague_terms and not has_context:
                logger.info("Detected ambiguous query (vague terms without academic context)",
                            question=question, 
                            has_vague_terms=has_vague_terms)
                return True

            return False

        except Exception as e:
            logger.error("Error in ambiguous query detection", error=str(e))
            return False

    async def handle_ambiguous_query(self, question: str) -> str:
        """Generate clarification for ambiguous queries with improved academic focus."""
        try:
            # Provide specific clarification based on the type of ambiguity
            question_lower = question.lower()

            # Check for queries needing clarification
            if re.search(r"table \d+|section \d+", question_lower):
                return """I need more specific information to answer your question about table/section data:

                1. **Which specific paper or document** are you referring to?
                2. **What specific metric, measure, or value** are you asking about?
                3. **What dataset or domain** are you interested in?
                4. **What specific aspect** of the table/section do you want to know?

                Please provide these details so I can give you precise information from the correct source."""

            elif "enough" in question_lower and ("examples" in question_lower or "data" in question_lower or "accuracy" in question_lower):
                return """"Enough" is vague—needs the dataset and the accuracy target. Please specify:

                1. **Which dataset or domain** are you referring to?
                2. **What accuracy target** do you consider "good"?
                3. **What specific context** (model type, task, baseline)?
                4. **What evaluation metric** should I focus on?

                Please provide these details so I can give you a precise answer based on the available research."""

            elif any(term in question_lower for term in ["good", "best", "optimal", "sufficient"]) and any(term in question_lower for term in ["accuracy", "performance", "result"]):
                return """To help you understand performance standards, I need clarification:

                1. **Which specific dataset or task** are you asking about?
                2. **What's your baseline** or comparison point?
                3. **What evaluation metric** are you interested in?
                4. **What's your specific use case or application context?**

                Please specify these details for a more helpful answer."""

            elif "model" in question_lower and any(term in question_lower for term in ["good", "best", "optimal"]) and "prediction" in question_lower:
                return """To help you find the right model for your prediction task, I need clarification:

                1. **What type of prediction** are you working on? (e.g., text-to-SQL, image classification, sentiment analysis)
                2. **What domain or dataset** are you using?
                3. **What performance criteria** are most important? (accuracy, speed, resource usage)
                4. **What's your specific use case** or application context?

                Please specify these details so I can provide relevant model recommendations from the available research."""

            elif "train" in question_lower and any(term in question_lower for term in ["long", "time", "duration"]):
                return """To help you understand training time requirements, I need clarification:

                1. **What type of model** are you referring to? (e.g., transformer, neural network, specific architecture)
                2. **What dataset size** are you working with? (number of examples, data volume)
                3. **What hardware setup** are you considering? (CPU, GPU, TPU, memory constraints)
                4. **What domain or task** is this for? (text-to-SQL, NLP, computer vision)
                5. **What performance target** do you need to achieve?

                Please specify these details so I can provide relevant training time estimates from the available research."""

            elif any(term in question_lower for term in ["successful", "success"]) and any(term in question_lower for term in ["long", "time", "take", "become"]):
                return """Your question about success is too general for this academic research database. I need clarification:

                1. **What specific domain** are you asking about? (e.g., machine learning, text-to-SQL, model training)
                2. **What type of success** do you mean? (e.g., achieving certain accuracy, completing training, reaching performance benchmarks)
                3. **What specific context** or task are you interested in?
                4. **What metrics** would define success in your case?

                Please provide more specific details so I can search the research papers for relevant information."""

            elif any(term in question_lower for term in ["good", "best", "right", "worth", "valuable", "fair", "enough", "much", "big", "fast", "safe", "smart", "easy", "special", "perfect", "strong", "close", "high", "real", "true"]):
                return """Your question contains subjective terms that need specific context. I need clarification:

                1. **What specific domain or task** are you asking about? (e.g., machine learning models, text-to-SQL systems, datasets, algorithms)
                2. **What criteria** would you use to measure these qualities? (e.g., accuracy metrics, performance benchmarks, specific requirements)
                3. **What's your specific use case** or application context?
                4. **What are you comparing against** or what's your baseline?

                Please provide these details so I can search the research papers for relevant quantitative information."""

            else:
                # General clarification prompt with academic focus
                clarification_prompt = f"""The user asked: "{question}"

                This question contains vague terms that need clarification. Generate 2-3 specific follow-up questions to help clarify the user's intent. Focus on identifying:
                - Specific datasets, metrics, or evaluation criteria mentioned or implied
                - Exact criteria for subjective terms that need quantification
                - Missing context about the specific use case, domain, or application
                - Any quantitative targets, thresholds, or performance goals
                - Specific sources, papers, or methodologies if referenced

                Format as a helpful clarification request that guides the user to provide more specific details."""

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
