"""
Conversational Query Processor for intelligent intent detection and query normalization.
Handles conversational input and determines if it's a legitimate dataset request.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm_clients import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class QueryProcessingResult:
    """Result of conversational query processing"""
    is_dataset_request: bool
    extracted_terms: List[str]
    confidence: float
    original_input: str
    suggested_clarification: Optional[str] = None
    detected_domain: Optional[str] = None
    requires_singapore_context: bool = False


class ConversationalQueryProcessor:
    """
    Processes conversational input to determine dataset intent and extract search terms.
    Integrates with existing LLMManager for intelligent intent detection.
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any]):
        """
        Initialize with existing LLM infrastructure
        
        Args:
            llm_manager: Existing LLMManager instance
            config: Configuration dictionary
        """
        self.llm_manager = llm_manager
        self.config = config
        self.query_config = config.get('conversational_query', {})
        
        # Intent detection settings
        self.confidence_threshold = self.query_config.get('confidence_threshold', 0.7)
        self.max_processing_time = self.query_config.get('max_processing_time', 3.0)
        
        # Singapore-specific keywords for context detection
        self.singapore_keywords = {
            'government': ['hdb', 'cpf', 'coe', 'lta', 'ura', 'singstat', 'moh', 'moe'],
            'transport': ['mrt', 'bus', 'taxi', 'grab', 'transport', 'traffic'],
            'housing': ['hdb', 'housing', 'property', 'resale', 'rental', 'bto'],
            'demographics': ['population', 'census', 'residents', 'citizens', 'pr'],
            'economy': ['gdp', 'inflation', 'employment', 'wages', 'economy', 'trade']
        }
        
        # Dataset-related keywords for intent detection
        self.dataset_keywords = [
            'data', 'dataset', 'statistics', 'information', 'records', 'database',
            'survey', 'census', 'report', 'analysis', 'research', 'study',
            'figures', 'numbers', 'metrics', 'indicators', 'trends'
        ]
        
        # Non-dataset patterns to filter out
        self.non_dataset_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon)\b',
            r'\b(how are you|what\'s up|what can you do)\b',
            r'\b(thank you|thanks|bye|goodbye)\b',
            r'\b(weather|news|joke|story|recipe)\b',
            r'\b(inappropriate|offensive|personal)\b'
        ]
        
        logger.info("ConversationalQueryProcessor initialized")
    
    async def process_input(self, user_input: str) -> QueryProcessingResult:
        """
        Process conversational input using existing LLM clients
        
        Args:
            user_input: Raw user input text
            
        Returns:
            QueryProcessingResult with intent analysis and extracted terms
        """
        start_time = time.time()
        logger.info(f"Processing input: '{user_input[:50]}...'")
        
        try:
            # Step 1: Quick rule-based filtering
            quick_result = self._quick_intent_check(user_input)
            if quick_result.confidence > 0.9:
                logger.info(f"Quick intent detection: {quick_result.is_dataset_request}")
                return quick_result
            
            # Step 2: LLM-based intent detection for ambiguous cases
            llm_result = await self._llm_intent_detection(user_input)
            
            # Step 3: Combine results and extract search terms
            final_result = self._combine_results(user_input, quick_result, llm_result)
            
            processing_time = time.time() - start_time
            logger.info(f"Intent processing completed in {processing_time:.2f}s: "
                       f"dataset_request={final_result.is_dataset_request}, "
                       f"confidence={final_result.confidence:.2f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Intent processing error: {e}")
            # Return safe fallback
            return QueryProcessingResult(
                is_dataset_request=False,
                extracted_terms=[],
                confidence=0.0,
                original_input=user_input,
                suggested_clarification="I'm having trouble understanding your request. Could you please specify what dataset or data you're looking for?"
            )
    
    def _quick_intent_check(self, user_input: str) -> QueryProcessingResult:
        """
        Quick rule-based intent detection for clear cases
        
        Args:
            user_input: User input text
            
        Returns:
            QueryProcessingResult with initial assessment
        """
        input_lower = user_input.lower().strip()
        
        # Check for non-dataset patterns
        for pattern in self.non_dataset_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return QueryProcessingResult(
                    is_dataset_request=False,
                    extracted_terms=[],
                    confidence=0.95,
                    original_input=user_input,
                    suggested_clarification=None
                )
        
        # Check for dataset keywords
        dataset_score = sum(1 for keyword in self.dataset_keywords if keyword in input_lower)
        
        # Check for Singapore context
        singapore_score = 0
        detected_domain = None
        for domain, keywords in self.singapore_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in input_lower)
            if domain_score > singapore_score:
                singapore_score = domain_score
                detected_domain = domain
        
        # Calculate confidence
        total_words = len(input_lower.split())
        dataset_confidence = min(dataset_score / max(total_words * 0.3, 1), 1.0)
        singapore_confidence = min(singapore_score / max(total_words * 0.2, 1), 1.0)
        
        # Determine if it's a dataset request
        is_dataset_request = dataset_confidence > 0.3 or singapore_confidence > 0.2
        confidence = max(dataset_confidence, singapore_confidence * 0.8)
        
        # Extract basic search terms
        extracted_terms = self._extract_basic_terms(user_input)
        
        return QueryProcessingResult(
            is_dataset_request=is_dataset_request,
            extracted_terms=extracted_terms,
            confidence=confidence,
            original_input=user_input,
            detected_domain=detected_domain,
            requires_singapore_context=singapore_confidence > 0.1
        )
    
    async def _llm_intent_detection(self, user_input: str) -> Dict[str, Any]:
        """
        Use LLM for sophisticated intent detection
        
        Args:
            user_input: User input text
            
        Returns:
            LLM analysis result
        """
        prompt = f"""Analyze this user input to determine if they're requesting dataset or data information:

User Input: "{user_input}"

Respond with a JSON object containing:
1. "is_dataset_request": true/false - Is this a request for datasets, data, statistics, or research information?
2. "confidence": 0.0-1.0 - How confident are you in this assessment?
3. "extracted_terms": ["term1", "term2"] - Key search terms for dataset queries (clean, no conversational words)
4. "domain": "category" - Data domain if applicable (housing, transport, demographics, economy, etc.)
5. "singapore_context": true/false - Does this relate to Singapore-specific data?
6. "clarification_needed": "question" - If ambiguous, what clarification question should we ask?

Examples:
- "I need HDB data" → dataset request, terms: ["HDB", "housing"], singapore: true
- "Hello, how are you?" → not dataset request
- "What's the weather?" → not dataset request  
- "Population statistics for Singapore" → dataset request, terms: ["population", "statistics"], singapore: true

Respond only with valid JSON."""

        try:
            result = await asyncio.wait_for(
                self.llm_manager.complete_with_fallback(
                    prompt=prompt,
                    preferred_provider="claude",
                    max_tokens=300,
                    temperature=0.3
                ),
                timeout=self.max_processing_time
            )
            
            # Parse JSON response
            content = result.get('content', '{}')
            return json.loads(content)
            
        except asyncio.TimeoutError:
            logger.warning("LLM intent detection timeout")
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")
            return {}
        except Exception as e:
            logger.warning(f"LLM intent detection error: {e}")
            return {}
    
    def _combine_results(self, 
                        user_input: str, 
                        quick_result: QueryProcessingResult, 
                        llm_result: Dict[str, Any]) -> QueryProcessingResult:
        """
        Combine rule-based and LLM results for final decision
        
        Args:
            user_input: Original user input
            quick_result: Rule-based analysis result
            llm_result: LLM analysis result
            
        Returns:
            Final QueryProcessingResult
        """
        # Use LLM result if available and confident
        if llm_result and llm_result.get('confidence', 0) > 0.7:
            return QueryProcessingResult(
                is_dataset_request=llm_result.get('is_dataset_request', False),
                extracted_terms=llm_result.get('extracted_terms', []),
                confidence=llm_result.get('confidence', 0.0),
                original_input=user_input,
                suggested_clarification=llm_result.get('clarification_needed'),
                detected_domain=llm_result.get('domain'),
                requires_singapore_context=llm_result.get('singapore_context', False)
            )
        
        # Fall back to rule-based result
        return quick_result
    
    def _extract_basic_terms(self, user_input: str) -> List[str]:
        """
        Extract basic search terms from user input
        
        Args:
            user_input: User input text
            
        Returns:
            List of extracted search terms
        """
        # Remove common conversational words
        stop_words = {
            'i', 'need', 'want', 'looking', 'for', 'find', 'get', 'show', 'me',
            'can', 'you', 'please', 'help', 'with', 'about', 'on', 'the', 'a', 'an',
            'and', 'or', 'but', 'in', 'at', 'to', 'from', 'of', 'is', 'are', 'was', 'were'
        }
        
        # Extract meaningful terms
        words = re.findall(r'\b\w+\b', user_input.lower())
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Keep only relevant terms (limit to 5 most important)
        return terms[:5]
    
    def extract_search_terms(self, conversational_input: str) -> List[str]:
        """
        Extract clean search terms from conversational input
        
        Args:
            conversational_input: Raw conversational input
            
        Returns:
            List of clean search terms suitable for external sources
        """
        return self._extract_basic_terms(conversational_input)
    
    def validate_dataset_intent(self, input_text: str) -> Tuple[bool, float]:
        """
        Determine if input is a legitimate dataset request
        
        Args:
            input_text: Input text to validate
            
        Returns:
            Tuple of (is_valid_request, confidence_score)
        """
        quick_result = self._quick_intent_check(input_text)
        return quick_result.is_dataset_request, quick_result.confidence
    
    def generate_clarification_prompt(self, ambiguous_input: str) -> str:
        """
        Generate clarification prompt for ambiguous inputs
        
        Args:
            ambiguous_input: The ambiguous user input
            
        Returns:
            Clarification question string
        """
        # Check what type of clarification is needed
        input_lower = ambiguous_input.lower()
        
        # Check for Singapore context first (more specific)
        if any(word in input_lower for word in ['singapore', 'sg']):
            return "Are you looking for Singapore-specific datasets? Please let me know what type of Singapore data you need (e.g., government data, economic indicators, demographic information)."
        
        if any(word in input_lower for word in ['data', 'information', 'statistics']):
            return "I can help you find datasets! Could you specify what type of data you're looking for? For example: housing data, transport statistics, population demographics, etc."
        
        return "I specialize in helping find datasets and research data. Could you please specify what type of dataset or data you're looking for?"
    
    def is_inappropriate_query(self, user_input: str) -> bool:
        """
        Check if query contains inappropriate content
        
        Args:
            user_input: User input to check
            
        Returns:
            True if inappropriate, False otherwise
        """
        inappropriate_patterns = [
            r'\b(hack|illegal|piracy|adult|explicit)\b',
            r'\b(personal|private|confidential|password)\b'
        ]
        
        input_lower = user_input.lower()
        return any(re.search(pattern, input_lower, re.IGNORECASE) for pattern in inappropriate_patterns)
    
    def enhance_query_with_context(self, 
                                 extracted_terms: List[str], 
                                 detected_domain: Optional[str] = None,
                                 singapore_context: bool = False) -> List[str]:
        """
        Enhance extracted terms with contextual information
        
        Args:
            extracted_terms: Basic extracted terms
            detected_domain: Detected data domain
            singapore_context: Whether Singapore context is relevant
            
        Returns:
            Enhanced search terms
        """
        enhanced_terms = extracted_terms.copy()
        
        # Add domain-specific terms
        if detected_domain and detected_domain in self.singapore_keywords:
            domain_terms = self.singapore_keywords[detected_domain][:2]  # Add top 2 domain terms
            enhanced_terms.extend([term for term in domain_terms if term not in enhanced_terms])
        
        # Add Singapore context if relevant
        if singapore_context and 'singapore' not in [term.lower() for term in enhanced_terms]:
            enhanced_terms.append('singapore')
        
        return enhanced_terms[:7]  # Limit to 7 terms total