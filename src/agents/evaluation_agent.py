"""Evaluation Agent for answer quality assessment and confidence scoring."""

from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime
import re

from ..core.models import QueryType, ChatMessage
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class EvaluationAgent:
    """Agent for evaluating answer quality and providing confidence scores."""

    def __init__(self):
        """Initialize the Evaluation Agent."""
        self.evaluation_history: List[Dict[str, Any]] = []
        # Configurable quality thresholds - can be adjusted based on requirements
        self.quality_thresholds = self._get_quality_thresholds()
        self.max_history = self._get_max_history_size()
        logger.info("Evaluation Agent initialized")

    def _get_quality_thresholds(self) -> Dict[str, float]:
        """Get configurable quality thresholds."""
        # These can be overridden based on environment or config
        from ..core.config import settings
        return getattr(settings, 'quality_thresholds', {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        })

    def _get_max_history_size(self) -> int:
        """Get configurable maximum history size."""
        from ..core.config import settings
        return getattr(settings, 'evaluation_history_limit', 1000)

    def _get_factual_indicators(self) -> List[str]:
        """Get configurable factual indicators."""
        from ..core.config import settings
        return getattr(settings, 'factual_indicators', [
            'according to', 'based on', 'research shows', 'studies indicate',
            'data suggests', 'evidence', 'findings show', 'results indicate',
            'analysis reveals', 'experiments demonstrate'
        ])

    def _get_domain_quality_scores(self) -> Dict[tuple, int]:
        """Get configurable domain quality scoring."""
        from ..core.config import settings
        return getattr(settings, 'domain_quality_scores', {
            ('.edu', '.gov', '.org'): 2,
            ('wikipedia', 'academic', 'research', 'scholar'): 1,
            ('arxiv', 'pubmed', 'ieee', 'acm'): 2,
            ('.com', '.net', 'blog'): 0
        })

    def _get_coherence_criteria(self) -> Dict[str, Any]:
        """Get configurable coherence evaluation criteria."""
        from ..core.config import settings
        return getattr(settings, 'coherence_criteria', {
            'min_length': 50,
            'max_length': 2000,
            'min_sentence_length': 10,
            'max_sentence_length': 100,
            'good_length_score': 1.0,
            'poor_length_score': 0.7,
            'good_sentence_score': 1.0,
            'poor_sentence_score': 0.8,
            'diversity_multiplier': 2
        })

    def _get_confidence_weights(self) -> Dict[str, float]:
        """Get configurable weights for overall confidence calculation."""
        from ..core.config import settings
        return getattr(settings, 'confidence_weights', {
            'relevance': 0.25,
            'completeness': 0.20,
            'coherence': 0.15,
            'factual_consistency': 0.20,
            'source_quality': 0.10,
            'retrieval_confidence': 0.10
        })

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        sources: List[str],
        query_type: QueryType,
        retrieval_confidence: float,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of an answer."""
        try:
            logger.info("Evaluating answer quality", question_length=len(
                question), answer_length=len(answer))

            # Run multiple evaluation metrics
            relevance_score = await self._evaluate_relevance(question, answer)
            completeness_score = await self._evaluate_completeness(question, answer, sources)
            coherence_score = await self._evaluate_coherence(answer)
            factual_consistency_score = await self._evaluate_factual_consistency(answer, sources, query_type)
            source_quality_score = self._evaluate_source_quality(
                sources, query_type)

            # Calculate overall confidence score
            overall_confidence = self._calculate_overall_confidence(
                relevance_score,
                completeness_score,
                coherence_score,
                factual_consistency_score,
                source_quality_score,
                retrieval_confidence
            )

            # Determine quality rating
            quality_rating = self._get_quality_rating(overall_confidence)

            # Generate evaluation summary
            evaluation_summary = await self._generate_evaluation_summary(
                question, answer, overall_confidence, quality_rating
            )

            evaluation_result = {
                'overall_confidence': round(overall_confidence, 3),
                'quality_rating': quality_rating,
                'detailed_scores': {
                    'relevance': round(relevance_score, 3),
                    'completeness': round(completeness_score, 3),
                    'coherence': round(coherence_score, 3),
                    'factual_consistency': round(factual_consistency_score, 3),
                    'source_quality': round(source_quality_score, 3),
                    'retrieval_confidence': round(retrieval_confidence, 3)
                },
                'evaluation_summary': evaluation_summary,
                'timestamp': datetime.now(),
                'query_type': query_type.value
            }

            # Store evaluation for analytics
            self._store_evaluation(question, answer, evaluation_result)

            logger.info(
                "Answer evaluation completed",
                overall_confidence=overall_confidence,
                quality_rating=quality_rating
            )

            return evaluation_result

        except Exception as e:
            logger.error("Error evaluating answer", error=str(e))
            return {
                'overall_confidence': retrieval_confidence,
                'quality_rating': 'unknown',
                'detailed_scores': {},
                'evaluation_summary': f"Evaluation failed: {str(e)}",
                'timestamp': datetime.now(),
                'query_type': query_type.value
            }

    async def _evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the question."""
        try:
            relevance_prompt = f"""Evaluate how well this answer addresses the given question on a scale of 0.0 to 1.0.

            Question: {question}

            Answer: {answer}

            Consider:
            - Does the answer directly address the question?
            - Is the answer on-topic?
            - Does it answer what was specifically asked?

            Provide only a decimal score between 0.0 and 1.0:"""

            score_text = await llm_service.generate_simple_response(relevance_prompt)
            
            # Handle empty or error responses
            if not score_text or score_text in ["Unable to generate response", "Error generating response"]:
                logger.warning("LLM returned empty/error response for relevance evaluation")
                return self._calculate_heuristic_relevance(question, answer)
            
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            else:
                logger.warning("Could not extract score from relevance evaluation", score_text=score_text[:200])
                return self._calculate_heuristic_relevance(question, answer)

        except Exception as e:
            logger.error("Error evaluating relevance", error=str(e))
            return 0.5  # Default moderate score

    async def _evaluate_completeness(self, question: str, answer: str, sources: List[str]) -> float:
        """Evaluate how complete the answer is."""
        try:
            completeness_prompt = f"""Evaluate how complete this answer is on a scale of 0.0 to 1.0.

            Question: {question}

            Answer: {answer}

            Number of sources used: {len(sources)}

            Consider:
            - Does the answer fully address all aspects of the question?
            - Is important information missing?
            - Is the answer comprehensive given the available sources?

            Provide only a decimal score between 0.0 and 1.0:"""

            score_text = await llm_service.generate_simple_response(completeness_prompt)
            
            # Handle empty or error responses
            if not score_text or score_text in ["Unable to generate response", "Error generating response"]:
                logger.warning("LLM returned empty/error response for completeness evaluation")
                return self._calculate_heuristic_completeness(question, answer, sources)
            
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            else:
                logger.warning("Could not extract score from completeness evaluation", score_text=score_text[:200])
                return self._calculate_heuristic_completeness(question, answer, sources)

        except Exception as e:
            logger.error("Error evaluating completeness", error=str(e))
            return 0.5

    async def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate the coherence and readability of the answer."""
        try:
            # Get configurable criteria
            criteria = self._get_coherence_criteria()

            # Basic coherence metrics
            sentences = answer.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = len(answer) / max(sentence_count, 1)

            # Check for repetition
            words = answer.lower().split()
            unique_words = len(set(words))
            word_diversity = unique_words / max(len(words), 1)

            # Configurable coherence score calculation
            length_score = (criteria['good_length_score']
                            if criteria['min_length'] <= len(answer) <= criteria['max_length']
                            else criteria['poor_length_score'])
            sentence_score = (criteria['good_sentence_score']
                              if criteria['min_sentence_length'] <= avg_sentence_length <= criteria['max_sentence_length']
                              else criteria['poor_sentence_score'])
            diversity_score = min(
                word_diversity * criteria['diversity_multiplier'], 1.0)

            coherence_score = (
                length_score + sentence_score + diversity_score) / 3

            return min(max(coherence_score, 0.0), 1.0)

        except Exception as e:
            logger.error("Error evaluating coherence", error=str(e))
            return 0.5

    async def _evaluate_factual_consistency(self, answer: str, sources: List[str], query_type: QueryType) -> float:
        """Evaluate factual consistency with sources."""
        try:
            if not sources:
                return 0.3  # Low score if no sources

            # Higher base score for PDF sources (more reliable)
            if query_type == QueryType.PDF_SEARCH:
                base_score = 0.8
            elif query_type == QueryType.WEB_SEARCH:
                base_score = 0.6
            else:
                base_score = 0.4

            # Check for factual claims in the answer
            factual_indicators = self._get_factual_indicators()

            has_factual_backing = any(indicator in answer.lower()
                                      for indicator in factual_indicators)
            if has_factual_backing:
                base_score += 0.1

            # Adjust based on number of sources
            source_bonus = min(len(sources) * 0.05, 0.2)

            final_score = min(base_score + source_bonus, 1.0)
            return final_score

        except Exception as e:
            logger.error("Error evaluating factual consistency", error=str(e))
            return 0.5

    def _evaluate_source_quality(self, sources: List[str], query_type: QueryType) -> float:
        """Evaluate the quality of sources used."""
        try:
            if not sources:
                return 0.0

            # PDF sources generally more reliable for document-specific queries
            if query_type == QueryType.PDF_SEARCH:
                base_score = 0.9  # PDF sources are curated
            elif query_type == QueryType.WEB_SEARCH:
                # Evaluate web source quality using configurable criteria
                quality_indicators = 0
                domain_scores = self._get_domain_quality_scores()

                for source in sources:
                    source_lower = source.lower()
                    # Check for reputable domains/indicators
                    for domain_group, score in domain_scores.items():
                        if any(domain in source_lower for domain in domain_group):
                            quality_indicators += score
                            break  # Only count the highest score per source

                base_score = min(0.5 + (quality_indicators * 0.1), 1.0)
            else:
                base_score = 0.3

            # Bonus for multiple sources
            diversity_bonus = min((len(sources) - 1) * 0.1, 0.3)

            return min(base_score + diversity_bonus, 1.0)

        except Exception as e:
            logger.error("Error evaluating source quality", error=str(e))
            return 0.5

    def _calculate_overall_confidence(
        self,
        relevance: float,
        completeness: float,
        coherence: float,
        factual_consistency: float,
        source_quality: float,
        retrieval_confidence: float
    ) -> float:
        """Calculate weighted overall confidence score."""
        try:
            # Weighted average with different importance
            weights = self._get_confidence_weights()

            overall_score = (
                relevance * weights['relevance'] +
                completeness * weights['completeness'] +
                coherence * weights['coherence'] +
                factual_consistency * weights['factual_consistency'] +
                source_quality * weights['source_quality'] +
                retrieval_confidence * weights['retrieval_confidence']
            )

            return min(max(overall_score, 0.0), 1.0)

        except Exception as e:
            logger.error("Error calculating overall confidence", error=str(e))
            return 0.5

    def _get_quality_rating(self, confidence_score: float) -> str:
        """Convert confidence score to quality rating."""
        if confidence_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif confidence_score >= self.quality_thresholds['good']:
            return 'good'
        elif confidence_score >= self.quality_thresholds['fair']:
            return 'fair'
        elif confidence_score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'

    async def _generate_evaluation_summary(
        self,
        question: str,
        answer: str,
        confidence: float,
        quality_rating: str
    ) -> str:
        """Generate a human-readable evaluation summary."""
        try:
            summary_prompt = f"""Generate a brief evaluation summary for this Q&A pair.

            Question: {question}
            Answer: {answer[:200]}...
            Confidence Score: {confidence:.3f}
            Quality Rating: {quality_rating}

            Provide a 1-2 sentence summary explaining the evaluation result:"""

            summary = await llm_service.generate_simple_response(summary_prompt)
            return summary.strip()

        except Exception as e:
            logger.error("Error generating evaluation summary", error=str(e))
            return f"Answer quality rated as {quality_rating} with {confidence:.1%} confidence."

    def _store_evaluation(self, question: str, answer: str, evaluation: Dict[str, Any]) -> None:
        """Store evaluation result for analytics."""
        try:
            evaluation_record = {
                'question': question,
                'answer': answer,
                'evaluation': evaluation,
                'timestamp': datetime.now()
            }

            self.evaluation_history.append(evaluation_record)

            # Keep only last N evaluations to prevent memory issues
            if len(self.evaluation_history) > self.max_history:
                self.evaluation_history = self.evaluation_history[-self.max_history:]

        except Exception as e:
            logger.error("Error storing evaluation", error=str(e))

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics and analytics."""
        try:
            if not self.evaluation_history:
                return {'message': 'No evaluations performed yet'}

            # Calculate statistics
            recent_evaluations = self.evaluation_history[-100:]  # Last 100

            confidences = [eval_record['evaluation']['overall_confidence']
                           for eval_record in recent_evaluations]

            quality_ratings = [eval_record['evaluation']['quality_rating']
                               for eval_record in recent_evaluations]

            # Quality distribution
            quality_distribution = {}
            for rating in quality_ratings:
                quality_distribution[rating] = quality_distribution.get(
                    rating, 0) + 1

            # Average scores by query type
            type_stats = {}
            for eval_record in recent_evaluations:
                query_type = eval_record['evaluation']['query_type']
                if query_type not in type_stats:
                    type_stats[query_type] = []
                type_stats[query_type].append(
                    eval_record['evaluation']['overall_confidence'])

            type_averages = {
                qtype: sum(scores) / len(scores)
                for qtype, scores in type_stats.items()
            }

            return {
                'total_evaluations': len(self.evaluation_history),
                'recent_evaluations': len(recent_evaluations),
                'average_confidence': sum(confidences) / len(confidences),
                'quality_distribution': quality_distribution,
                'confidence_by_query_type': type_averages,
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            }

        except Exception as e:
            logger.error("Error getting evaluation statistics", error=str(e))
            return {'error': str(e)}

    def _calculate_heuristic_relevance(self, question: str, answer: str) -> float:
        """Calculate relevance score using heuristic methods when LLM fails."""
        try:
            if not answer or answer.strip() in ["Unable to generate response", "Error generating response", ""]:
                return 0.1  # Low score for empty/error responses
            
            # Basic word overlap scoring
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(question_words.intersection(answer_words))
            overlap_score = min(overlap / max(len(question_words), 1), 1.0)
            
            # Length penalty for very short answers
            length_factor = min(len(answer) / 100, 1.0)  # Penalize answers shorter than 100 chars
            
            final_score = (overlap_score * 0.7 + length_factor * 0.3)
            return min(max(final_score, 0.1), 0.8)  # Cap at 0.8 for heuristic
            
        except Exception as e:
            logger.error("Error in heuristic relevance calculation", error=str(e))
            return 0.5

    def _calculate_heuristic_completeness(self, question: str, answer: str, sources: List[str]) -> float:
        """Calculate completeness score using heuristic methods when LLM fails."""
        try:
            if not answer or answer.strip() in ["Unable to generate response", "Error generating response", ""]:
                return 0.1  # Low score for empty/error responses
            
            # Base score from answer length
            length_score = min(len(answer) / 500, 1.0)  # Normalize to 500 chars
            
            # Bonus for having sources
            source_score = min(len(sources) / 5, 1.0)  # Up to 5 sources is good
            
            # Bonus for structured content (lists, numbers, etc.)
            structure_bonus = 0.1 if any(marker in answer for marker in ['\n-', '\n1.', '\n•', ':', ';']) else 0
            
            final_score = (length_score * 0.5 + source_score * 0.3 + structure_bonus * 0.2)
            return min(max(final_score, 0.1), 0.8)  # Cap at 0.8 for heuristic
            
        except Exception as e:
            logger.error("Error in heuristic completeness calculation", error=str(e))
            return 0.5

    def clear_evaluation_history(self) -> bool:
        """Clear evaluation history."""
        try:
            self.evaluation_history.clear()
            logger.info("Evaluation history cleared")
            return True
        except Exception as e:
            logger.error("Error clearing evaluation history", error=str(e))
            return False


# Global evaluation agent instance
evaluation_agent = EvaluationAgent()
