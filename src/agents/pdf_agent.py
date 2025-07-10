"""PDF Agent for RAG-based document retrieval and question answering."""

from typing import List, Dict, Any, Tuple
import structlog
import re

from ..core.models import ChatMessage
from ..services.vector_store import vector_store_service
from ..services.llm_service import llm_service

logger = structlog.get_logger(__name__)


class PDFAgent:
    """Agent for answering questions using PDF documents via RAG."""

    def __init__(self):
        """Initialize the PDF Agent."""
        self.max_retrieved_docs = 15  # Increased to capture more diverse results
        # Lowered to capture more diverse results including table data
        self.score_threshold = 0.05
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

            # Search for relevant documents with enhanced strategy
            retrieved_docs = await self._enhanced_document_search(question)

            if not retrieved_docs:
                no_docs_response = (
                    "I couldn't find any relevant information in the available PDF documents "
                    "to answer your question. The documents might not contain information "
                    "about this topic, or you might want to try rephrasing your question."
                )
                return no_docs_response, [], 0.0

            # Prepare context from retrieved documents with enhanced table handling
            context_parts = []
            sources = []

            for i, doc in enumerate(retrieved_docs):
                content = doc['content']
                metadata = doc['metadata']
                score = doc['score']

                # Enhanced content processing for table data
                processed_content = self._enhance_table_content(
                    content, question)

                # Add document context
                source_info = f"{metadata.get('source', 'Unknown')} (Page {metadata.get('page', 'N/A')})"
                context_parts.append(
                    f"Document {i+1} ({source_info}):\n{processed_content}")

                # Track unique sources
                if source_info not in sources:
                    sources.append(source_info)

            # Combine context
            context = "\n\n".join(context_parts)

            # Generate answer using LLM with enhanced prompt for numerical data
            answer = await self._generate_enhanced_answer(
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

    def _enhance_table_content(self, content: str, question: str) -> str:
        """Enhanced content processing for table data and numerical information."""
        try:
            # Check if this is a table or numerical chunk
            if content.startswith('TABLE DATA:') or content.startswith('NUMERICAL DATA:'):
                # This is a special chunk, extract the actual data
                actual_content = content.replace('TABLE DATA:', '').replace(
                    'NUMERICAL DATA:', '').strip()

                # For table data, try to format it better
                if content.startswith('TABLE DATA:'):
                    # Look for percentage patterns and format them
                    actual_content = re.sub(
                        r'(\d+\.?\d*%)\s*', r'\1 ', actual_content)
                    actual_content = re.sub(
                        r'(\d+\.?\d*)\s*%\s*', r'\1% ', actual_content)

                return actual_content

            # Regular content processing
            # Look for percentage patterns and ensure they're properly formatted
            content = re.sub(r'(\d+\.?\d*%)\s*', r'\1 ', content)
            content = re.sub(r'(\d+\.?\d*)\s*%\s*', r'\1% ', content)

            # Look for number patterns mentioned in the question
            question_numbers = re.findall(r'\d+\.?\d*', question)
            for number in question_numbers:
                if number in content:
                    # Highlight this number in the content
                    content = re.sub(
                        rf'\b({re.escape(number)})\b', r'**\1**', content)

            return content

        except Exception as e:
            logger.error("Error enhancing table content", error=str(e))
            return content

    async def _generate_enhanced_answer(
        self,
        question: str,
        context: str,
        sources: List[str],
        chat_history: List[ChatMessage] = None
    ) -> str:
        """Generate enhanced answer with better handling of numerical data and table information."""
        try:
            # Check if this is a numerical/table query
            is_numerical_query = any(term in question.lower() for term in [
                'percentage', 'percent', '%', 'accuracy', 'score', 'number', 'value',
                'execution', 'exact match', 'f1', 'table', 'figure', 'data'
            ]) or any(char.isdigit() for char in question)

            # Check for error analysis queries specifically
            is_error_analysis_query = any(term in question.lower() for term in [
                'error', 'analysis', 'failure', 'mistake', 'wrong', 'incorrect'
            ])

            # Check for queries about comparative performance or rankings
            is_prompt_accuracy_query = any(term in question.lower() for term in [
                'highest', 'best', 'top', 'optimal', 'maximum', 'superior',
                'template', 'method', 'approach', 'technique'
            ]) and any(term in question.lower() for term in [
                'accuracy', 'performance', 'score', 'result'
            ])

            if is_error_analysis_query:
                # Specific prompt for error analysis to prevent fabrication
                enhanced_prompt = f"""Answer the question about error analysis using only the information explicitly documented in the provided context.

                Question: {question}

                Context from research papers:
                {context}

                Instructions:
                - Describe ONLY the error types, examples, and findings that are explicitly mentioned in the context
                - NEVER generate specific percentages or numerical breakdowns like "25.2%" unless they appear in the provided text
                - Focus on documented error examples, patterns, and conclusions from the authors
                - If the context mentions general performance metrics (like overall accuracy), include only those exact numbers
                - Describe error categories only if they are named and described in the context
                - Keep the explanation brief and focused on documented findings
                - Do NOT fabricate statistical breakdowns or categorization percentages

                Format: Provide a brief explanation using only the documented error analysis information."""

                return await llm_service.generate_simple_response(enhanced_prompt)

            elif is_prompt_accuracy_query:
                # Conservative prompt for best approach/method queries
                enhanced_prompt = f"""Answer the question about the best approach or method based strictly on what is documented in the provided context.

                Question: {question}

                Context from research papers:
                {context}

                Instructions:
                - Extract ONLY the numerical data and method names that are explicitly stated in the context
                - NEVER infer, calculate, or generate percentages not directly written in the text
                - Present only the performance comparisons that are clearly documented
                - If specific numerical breakdowns are not in the context, describe findings generally without fabricating numbers
                - Focus on what the authors explicitly conclude about methods and performance
                - Do not create detailed categorizations unless they appear verbatim in the context
                - CRITICAL: Do not fabricate statistics like "25.2%" or "14 E%" - only use numbers that actually appear in the text

                Format: Present the documented findings using only the data explicitly provided in the context."""

                response = await llm_service.generate_simple_response(enhanced_prompt)
                if not response or not response.strip():
                    logger.warning("Empty response from LLM for prompt accuracy query")
                    # Fallback to standard answer generation
                    return await llm_service.synthesize_answer(
                        question=question,
                        context=context,
                        sources=sources,
                        chat_history=chat_history
                    )
                return response

            elif is_numerical_query:
                # Conservative prompt for numerical data
                enhanced_prompt = f"""Answer the question about numerical data using only what is explicitly stated in the provided context.
                
                Question: {question}
                
                Context from research papers:
                {context}
                
                Instructions:
                - Report ONLY numerical values that are directly written in the context
                - NEVER calculate, infer, or generate any percentages not explicitly stated
                - Quote exact numbers and metrics as they appear in the text
                - If detailed breakdowns are not provided, summarize what is actually documented without adding fake numbers
                - Focus on the specific numerical data the authors present
                - Do not create error categories or percentages unless they appear verbatim
                - CRITICAL: If you cannot find specific percentages in the context, do NOT make them up
                
                Format: Provide a very brief 1-2 sentence summary."""

                response = await llm_service.generate_simple_response(enhanced_prompt)
                if not response or not response.strip():
                    logger.warning("Empty response from LLM for numerical query")
                    # Fallback to standard answer generation
                    return await llm_service.synthesize_answer(
                        question=question,
                        context=context,
                        sources=sources,
                        chat_history=chat_history
                    )
                return response
            else:
                # Standard answer generation
                return await llm_service.synthesize_answer(
                    question=question,
                    context=context,
                    sources=sources,
                    chat_history=chat_history
                )

        except Exception as e:
            logger.error("Error generating enhanced answer", error=str(e))
            # Fallback to standard answer generation
            return await llm_service.synthesize_answer(
                question=question,
                context=context,
                sources=sources,
                chat_history=chat_history
            )

    async def _enhanced_document_search(self, question: str) -> List[Dict[str, Any]]:
        """Enhanced document search that tries multiple strategies to find relevant content."""
        try:
            # First, try the regular search
            docs = await self.search_documents(question)

            # Extract paper information from the question - multiple patterns
            # Pattern 1: "...in [Author] [Year]" or "...in [Author] et al. [Year]"
            paper_match = re.search(
                r'\bin\s+([A-Z][a-z]+(?:\s+(?:and|et\s+al\.?))?\s+[A-Z][a-z]*(?:\s+et\s+al\.?)?\s*\(?\d{4}\)?)', question)

            # Pattern 2: "[Author] et al. - [Year]" or "[Author] et al. [Year]" anywhere in the question
            author_match = re.search(r'([A-Z][a-z]+)\s+et\s+al\.?\s*-?\s*(\d{4})', question) or \
                re.search(r'([A-Z][a-z]+)\s+et\s+al\.?\s*\((\d{4})\)', question) or \
                re.search(r'([A-Z][a-z]+)(?:\s+(?:and|et\s+al\.?))?\s+[A-Z][a-z]*(?:\s+et\s+al\.?)?\s*\((\d{4})\)', question) or \
                re.search(
                    r'([A-Z][a-z]+)(?:\s+and\s+[A-Z][a-z]+)?\s*.*?(\d{4})', question)

            # Handle paper-specific search using "in [Paper Reference]" pattern
            if paper_match:
                paper_reference = paper_match.group(1)
                logger.info(f"Found paper reference: {paper_reference}")

                # Search directly for the paper reference
                paper_docs = await self.search_documents(paper_reference, top_k=10, score_threshold=0.05)
                docs.extend(paper_docs)

                # Extract author and year from the paper reference
                ref_author_match = re.search(r'([A-Z][a-z]+)', paper_reference)
                ref_year_match = re.search(r'(\d{4})', paper_reference)

                if ref_author_match and ref_year_match:
                    author_name = ref_author_match.group(1)
                    year = ref_year_match.group(1)

                    # Filter for documents from this specific paper
                    paper_specific_docs = [
                        doc for doc in docs
                        if author_name.lower() in doc['metadata'].get('source', '').lower() and
                        year in doc['metadata'].get('source', '')
                    ]

                    # If we found paper-specific docs, prioritize them heavily
                    if paper_specific_docs:
                        logger.info(
                            f"Found {len(paper_specific_docs)} documents from {author_name} {year}")
                        for doc in paper_specific_docs:
                            # Triple boost for exact paper match
                            doc['score'] = doc['score'] * 3.0
                        docs.extend(paper_specific_docs)

            elif author_match:
                author_name = author_match.group(1)
                year = author_match.group(2)

                # Check if we have results from the mentioned author
                has_author_results = any(
                    author_name.lower() in doc['metadata'].get('source', '').lower() and
                    year in doc['metadata'].get('source', '')
                    for doc in docs
                )

                if not has_author_results:
                    logger.info(
                        f"No results found for {author_name} {year}, trying targeted search")

                    # Try a more targeted search for the specific author/year
                    targeted_query = f"{author_name} {year}"
                    targeted_docs = await self.search_documents(targeted_query, top_k=5, score_threshold=0.05)

                    # Extract key terms dynamically from the question
                    key_terms = []
                    question_words = question.lower().split()

                    # Add relevant nouns and technical terms from the question
                    technical_terms = ['template', 'accuracy',
                                       'performance', 'method', 'approach', 'model']
                    for word in question_words:
                        if word in technical_terms or word.endswith('ing') or word.endswith('ed'):
                            key_terms.append(word)

                    # Add any numbers found in the question
                    for word in question_words:
                        if any(char.isdigit() for char in word):
                            key_terms.append(word)

                    # Add specific error analysis terms if mentioned
                    error_analysis_terms = [
                        'error', 'analysis', 'failure', 'mistake', 'incorrect', 'wrong', 'limitation', 'challenge']
                    for word in question_words:
                        if word in error_analysis_terms:
                            key_terms.append(word)
                            # Also add common error analysis combinations
                            if word == 'error' or word == 'analysis':
                                key_terms.extend(
                                    ['error analysis', 'failure analysis', 'error patterns'])

                    for term in key_terms:
                        term_docs = await self.search_documents(term, top_k=8, score_threshold=0.05)
                        # Filter for the specific author
                        author_term_docs = [
                            doc for doc in term_docs
                            if author_name.lower() in doc['metadata'].get('source', '').lower() and
                            year in doc['metadata'].get('source', '')
                        ]
                        docs.extend(author_term_docs)
                        logger.info(
                            f"Found {len(author_term_docs)} docs for term '{term}' from {author_name} {year}")

                    # Add targeted docs
                    docs.extend(targeted_docs)

            # Enhanced search for comparative or performance-related content
            comparative_terms = ['highest', 'best', 'optimal',
                                 'superior', 'maximum', 'compare', 'comparison']
            if any(term in question.lower() for term in comparative_terms):
                # Search for tables and figures that might contain comparative data
                for table_num in range(1, 6):  # Search tables 1-5
                    table_docs = await self.search_documents(f'Table {table_num}', top_k=5, score_threshold=0.05)
                    docs.extend(table_docs)

                # Search for performance-related terms
                performance_terms = [
                    'outperforms', 'performance', 'results', 'comparison', 'evaluation']
                for term in performance_terms:
                    if term in question.lower():
                        perf_docs = await self.search_documents(term, top_k=6, score_threshold=0.05)
                        docs.extend(perf_docs)
                        logger.info(f"Added {len(perf_docs)} {term} documents")

            # Enhanced search for table-specific content
            if any(term in question.lower() for term in ['table', 'figure', 'data', 'results', 'percentage', '%']):
                # Search for table chunks specifically
                table_docs = await self.search_documents('TABLE DATA', top_k=8, score_threshold=0.05)
                docs.extend(table_docs)
                logger.info(f"Added {len(table_docs)} table-related documents")

            # Enhanced search for numerical data
            numerical_terms = ['percentage', '%', 'accuracy',
                               'score', 'number', 'value', 'result']
            has_numbers = any(char.isdigit() for char in question)

            if any(term in question.lower() for term in numerical_terms) or has_numbers:
                # Search for numerical chunks specifically
                numerical_docs = await self.search_documents('NUMERICAL DATA', top_k=8, score_threshold=0.05)
                docs.extend(numerical_docs)
                logger.info(
                    f"Added {len(numerical_docs)} numerical data documents")

                # Extract numbers from the question and search for them
                numbers = re.findall(r'\d+\.?\d*', question)
                for number in numbers:
                    number_docs = await self.search_documents(number, top_k=5, score_threshold=0.05)
                    docs.extend(number_docs)
                    logger.info(
                        f"Added {len(number_docs)} documents for number {number}")

            # Remove duplicates based on content (keep highest scoring version)
            seen_content = {}

            for doc in docs:
                # Use first 100 chars as key
                content_key = doc['content'][:100]
                if content_key not in seen_content or doc['score'] > seen_content[content_key]['score']:
                    seen_content[content_key] = doc

            unique_docs = list(seen_content.values())

            # If we found an author mentioned in the question, prioritize their results
            if author_match:
                author_docs = []
                other_docs = []

                for doc in unique_docs:
                    source = doc['metadata'].get('source', '')
                    if author_name.lower() in source.lower() and year in source:
                        # Boost the score significantly for mentioned author's content
                        # 100% boost for exact match
                        doc['score'] = doc['score'] * 2.0
                        author_docs.append(doc)
                    else:
                        other_docs.append(doc)

                # Sort each group by score
                author_docs.sort(key=lambda x: x['score'], reverse=True)
                other_docs.sort(key=lambda x: x['score'], reverse=True)

                # Prioritize author docs heavily - use 80% of slots for mentioned author
                max_author_docs = min(
                    len(author_docs), int(self.max_retrieved_docs * 0.8))
                max_other_docs = self.max_retrieved_docs - max_author_docs

                final_docs = author_docs[:max_author_docs] + \
                    other_docs[:max_other_docs]
                return final_docs[:self.max_retrieved_docs]
            else:
                # Normal sorting if no specific author mentioned
                unique_docs.sort(key=lambda x: x['score'], reverse=True)
                return unique_docs[:self.max_retrieved_docs]

        except Exception as e:
            logger.error("Error in enhanced document search", error=str(e))
            # Fallback to regular search
            return await self.search_documents(question)

    async def _is_out_of_scope_query(self, question: str) -> bool:
        """Check if a query is clearly out of scope for academic papers."""
        try:
            question_lower = question.lower()

            # Check for temporal indicators suggesting current events
            temporal_patterns = [
                r'\btoday\b', r'\byesterday\b', r'\bthis (week|month|year)\b',
                r'\brecently\b', r'\blatest\b', r'\bjust (announced|released)\b',
                r'\bcurrent\b', r'\bnow\b', r'\bbreaking\b'
            ]

            # Check for corporate/commercial contexts
            commercial_patterns = [
                r'\b(company|corporation|startup|business)\b.*\b(announce|release|launch)\b',
                r'\b(announce|release|launch)\b.*\b(company|corporation|startup|business)\b'
            ]

            import re
            # Check temporal patterns
            has_temporal = any(re.search(pattern, question_lower)
                               for pattern in temporal_patterns)

            # Check commercial patterns
            has_commercial = any(re.search(pattern, question_lower)
                                 for pattern in commercial_patterns)

            # Out of scope if clearly temporal AND commercial context
            if has_temporal and has_commercial:
                logger.info("Detected out-of-scope query", question=question)
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
