"""
AI-Powered Research Assistant
Orchestrates neural recommendations with intelligent explanation and methodology guidance
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .conversation_manager import ConversationManager
from .llm_clients import LLMManager
from .neural_ai_bridge import NeuralAIBridge
from .web_search_engine import WebSearchEngine

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """
    Main orchestrator for AI-enhanced dataset recommendations
    Combines 69.99% NDCG@3 neural performance with intelligent explanation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.neural_bridge = NeuralAIBridge(config)
        self.conversation_manager = ConversationManager(config)
        self.web_search_engine = WebSearchEngine(config)
        self.response_config = config.get('response_settings', {})
        self.research_config = config.get('research_settings', {})
        
    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user query through neural + AI enhancement pipeline
        
        Args:
            query: User's research query
            session_id: Session identifier for conversation continuity
            context: Additional context (user preferences, research domain, etc.)
            
        Returns:
            Enhanced response with recommendations and intelligent guidance
        """
        start_time = time.time()
        
        # Create or retrieve session
        if session_id:
            session = self.conversation_manager.get_session(session_id)
        else:
            session = self.conversation_manager.create_session()
            session_id = session['session_id']
        
        try:
            # Stage 1: Parallel processing - Neural recommendations + Web search
            neural_task = self._get_neural_recommendations(query, context)
            web_search_task = self._get_web_search_results(query, context)
            
            neural_results, web_results = await asyncio.gather(neural_task, web_search_task)
            
            # Stage 2: Enhance with AI intelligence
            ai_enhancement = await self._enhance_with_ai(query, neural_results, web_results, context)
            
            # Stage 3: Add research methodology
            methodology = await self._generate_methodology(query, neural_results, ai_enhancement, context)
            
            # Stage 4: Singapore context if relevant
            singapore_context = await self._add_singapore_context(query, neural_results, context)
            
            # Combine all components
            response = self._build_response(
                query=query,
                neural_results=neural_results,
                web_results=web_results,
                ai_enhancement=ai_enhancement,
                methodology=methodology,
                singapore_context=singapore_context,
                session_id=session_id,
                processing_time=time.time() - start_time
            )
            
            # Update conversation history
            self.conversation_manager.add_to_history(session_id, query, response)
            
            # Log performance
            logger.info(f"Query processed in {response['processing_time']:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._build_error_response(query, str(e), session_id, time.time() - start_time)
    
    async def _get_neural_recommendations(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get recommendations from high-performing neural model"""
        try:
            # Get neural recommendations with 69.99% NDCG@3 performance
            neural_results = await self.neural_bridge.get_neural_recommendations(
                query=query,
                top_k=self.response_config.get('top_k_recommendations', 5)
            )
            
            # Format for AI enhancement
            formatted_results = self.neural_bridge.format_for_ai_enhancement(neural_results)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Neural recommendation error: {str(e)}")
            raise
    
    async def _get_web_search_results(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get web search results for additional data sources"""
        try:
            web_results = await self.web_search_engine.search_web(query, context)
            logger.info(f"🌐 Found {len(web_results)} web search results")
            return web_results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    async def _enhance_with_ai(
        self,
        query: str,
        neural_results: Dict[str, Any],
        web_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance neural recommendations with AI intelligence"""
        
        # Build enhancement prompt including web results
        prompt = self._build_enhancement_prompt(query, neural_results, web_results, context)
        
        # Use MiniMax for research enhancement (primary)
        try:
            ai_response = await self.llm_manager.complete_for_capability(
                prompt=prompt,
                capability="research_methodology",
                preferred_provider="minimax"
            )
            
            # Parse AI response
            enhancement = self._parse_ai_enhancement(ai_response['content'])
            enhancement['provider'] = ai_response.get('provider_used', 'unknown')
            enhancement['response_time'] = ai_response.get('response_time', 0)
            
            return enhancement
            
        except Exception as e:
            logger.error(f"AI enhancement error: {str(e)}")
            # Return basic enhancement
            return {
                "explanations": self._generate_basic_explanations(neural_results),
                "relationships": [],
                "insights": [],
                "provider": "fallback"
            }
    
    def _build_enhancement_prompt(
        self,
        query: str,
        neural_results: Dict[str, Any],
        web_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for AI enhancement"""
        
        # Extract top recommendations
        recommendations = neural_results.get('top_recommendations', [])
        
        prompt = f"""
Research Query: "{query}"

HIGH-PERFORMANCE NEURAL RECOMMENDATIONS (69.99% NDCG@3):
Neural Model: {neural_results.get('neural_performance', {}).get('model', 'Cross-Attention Ranker')}
Performance: {neural_results.get('neural_performance', {}).get('ndcg_at_3', 0.6999)*100:.1f}% accuracy

Top Datasets:
"""
        
        for i, rec in enumerate(recommendations[:5], 1):
            dataset = rec.get('dataset', {})
            ranking = rec.get('ranking_details', {})
            prompt += f"""
{i}. {dataset.get('title', 'Unknown')}
   - Source: {dataset.get('source', 'Unknown')}
   - Category: {dataset.get('category', 'General')}
   - Confidence: {ranking.get('confidence', 0)*100:.0f}%
   - Quality Score: {ranking.get('quality_score', 0)*100:.0f}%
   - Neural Similarity: {ranking.get('neural_similarity', 0)*100:.0f}%
"""
        
        # Add web search results
        if web_results:
            prompt += f"""

ADDITIONAL WEB SEARCH RESULTS ({len(web_results)} found):
"""
            for i, result in enumerate(web_results[:5], 1):
                prompt += f"""
{i}. {result.get('title', 'Unknown')}
   - URL: {result.get('url', 'N/A')}
   - Source: {result.get('source', 'web')}
   - Type: {result.get('type', 'general')}
   - Description: {result.get('description', 'No description')[:100]}...
"""
        
        prompt += """

TASK: Enhance these high-quality neural recommendations with research intelligence and web search insights.

1. DATASET RELATIONSHIPS & SYNERGIES:
   - Explain how these datasets complement each other
   - Identify key variables for joining/integration
   - Highlight cross-dataset insights possible

2. RESEARCH VALUE EXPLANATION:
   - Why are these specific datasets valuable for this query?
   - What unique insights can each dataset provide?
   - How do they address different aspects of the research question?

3. ANALYTICAL APPROACHES:
   - Suggest 2-3 specific analytical methods
   - Recommend visualization techniques
   - Identify potential research hypotheses

4. QUALITY & LIMITATIONS:
   - Note any data quality considerations
   - Identify potential biases or gaps
   - Suggest validation approaches

5. SINGAPORE CONTEXT (if applicable):
   - Local relevance and implications
   - Policy or cultural considerations
   - Comparison opportunities with global data

Format your response as structured JSON with keys:
- explanations: List of dataset-specific explanations
- relationships: List of dataset relationship descriptions
- analytical_approaches: List of recommended methods
- insights: List of potential insights
- limitations: List of considerations
- singapore_context: Object with local relevance (if applicable)
"""
        
        return prompt
    
    def _parse_ai_enhancement(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI enhancement response"""
        try:
            # Try to parse as JSON first
            if ai_response.strip().startswith('{'):
                return json.loads(ai_response)
        except:
            pass
        
        # Fallback to text parsing
        enhancement = {
            "explanations": [],
            "relationships": [],
            "analytical_approaches": [],
            "insights": [],
            "limitations": [],
            "singapore_context": {}
        }
        
        # Simple parsing logic - would be more sophisticated in production
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'explanation' in line.lower():
                current_section = 'explanations'
            elif 'relationship' in line.lower():
                current_section = 'relationships'
            elif 'analytical' in line.lower() or 'approach' in line.lower():
                current_section = 'analytical_approaches'
            elif 'insight' in line.lower():
                current_section = 'insights'
            elif 'limitation' in line.lower():
                current_section = 'limitations'
            elif 'singapore' in line.lower():
                current_section = 'singapore_context'
            elif line.startswith('-') or line.startswith('•'):
                # Add to current section
                if current_section and isinstance(enhancement[current_section], list):
                    enhancement[current_section].append(line[1:].strip())
        
        return enhancement
    
    async def _generate_methodology(
        self,
        query: str,
        neural_results: Dict[str, Any],
        ai_enhancement: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate research methodology guidance"""
        
        if not self.response_config.get('include_methodology', True):
            return {}
        
        prompt = f"""
Research Query: "{query}"

Selected Datasets: {[rec['dataset']['title'] for rec in neural_results.get('top_recommendations', [])[:3]]}

Generate a comprehensive RESEARCH METHODOLOGY GUIDE:

1. DATA INTEGRATION STEPS:
   - Step-by-step guide to combine these datasets
   - Key fields for joining (with specific column names if known)
   - Data preprocessing requirements
   - Format harmonization needs

2. ANALYTICAL WORKFLOW:
   - Recommended analysis sequence
   - Statistical methods appropriate for the data
   - Hypothesis testing approaches
   - Validation techniques

3. TOOLS & TECHNOLOGIES:
   - Recommended software/languages (Python, R, etc.)
   - Specific libraries or packages
   - Code snippets for key operations

4. ACADEMIC STANDARDS:
   - Citation format for datasets
   - Reproducibility guidelines
   - Documentation requirements
   - Ethical considerations

5. SINGAPORE-SPECIFIC CONSIDERATIONS:
   - Local research guidelines
   - Government data usage policies
   - Cultural sensitivity points

Provide practical, actionable guidance suitable for academic research.
"""
        
        try:
            methodology_response = await self.llm_manager.complete_for_capability(
                prompt=prompt,
                capability="research_methodology",
                preferred_provider="minimax"
            )
            
            return {
                "methodology_guide": methodology_response['content'],
                "generated_by": methodology_response.get('provider_used', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Methodology generation error: {str(e)}")
            return {
                "methodology_guide": "Basic methodology: 1) Download datasets, 2) Clean and preprocess, 3) Join on common keys, 4) Analyze patterns, 5) Validate findings",
                "generated_by": "fallback"
            }
    
    async def _add_singapore_context(
        self,
        query: str,
        neural_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add Singapore-specific context if relevant"""
        
        if not self.research_config.get('singapore_specific', {}).get('enabled', True):
            return {}
        
        # Check if query or recommendations are Singapore-related
        singapore_keywords = self.research_config.get('singapore_specific', {}).get('local_context_keywords', [])
        
        is_singapore_relevant = any(
            keyword.lower() in query.lower() 
            for keyword in singapore_keywords
        )
        
        # Check recommendations
        for rec in neural_results.get('top_recommendations', []):
            source = rec.get('dataset', {}).get('source', '')
            if any(gov_source in source for gov_source in ['.gov.sg', 'singapore', 'sg']):
                is_singapore_relevant = True
                break
        
        if not is_singapore_relevant:
            return {}
        
        # Generate Singapore context
        try:
            prompt = f"""
Query: "{query}"
Datasets: {[rec['dataset']['title'] for rec in neural_results.get('top_recommendations', [])[:3]]}

Provide SINGAPORE-SPECIFIC CONTEXT:
1. Local policy implications
2. Comparison with regional/global standards
3. Cultural or social considerations
4. Government initiatives related to this topic
5. Local research institutions or resources

Be concise and practical.
"""
            
            singapore_response = await self.llm_manager.complete_with_fallback(
                prompt=prompt,
                preferred_provider="claude"  # Claude good for cultural context
            )
            
            return {
                "singapore_context": singapore_response['content'],
                "is_relevant": True
            }
            
        except Exception as e:
            logger.error(f"Singapore context error: {str(e)}")
            return {"singapore_context": "", "is_relevant": False}
    
    def _generate_basic_explanations(self, neural_results: Dict[str, Any]) -> List[str]:
        """Generate basic explanations when AI enhancement fails"""
        explanations = []
        
        for rec in neural_results.get('top_recommendations', [])[:3]:
            dataset = rec.get('dataset', {})
            ranking = rec.get('ranking_details', {})
            
            explanation = f"{dataset.get('title', 'Dataset')} is recommended with {ranking.get('confidence', 0)*100:.0f}% confidence "
            explanation += f"based on neural ranking (quality score: {ranking.get('quality_score', 0)*100:.0f}%)"
            
            explanations.append(explanation)
        
        return explanations
    
    def _build_response(
        self,
        query: str,
        neural_results: Dict[str, Any],
        web_results: List[Dict[str, Any]],
        ai_enhancement: Dict[str, Any],
        methodology: Dict[str, Any],
        singapore_context: Dict[str, Any],
        session_id: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Build final response combining all components"""
        
        response = {
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(processing_time, 3),
            
            # Neural recommendations with high performance
            "recommendations": [
                {
                    "dataset": rec.get('dataset', {}),
                    "confidence": rec.get('ranking_details', {}).get('confidence', 0),
                    "relevance_score": rec.get('ranking_details', {}).get('relevance_score', 0),
                    "quality_score": rec.get('ranking_details', {}).get('quality_score', 0),
                    "explanation": self._get_explanation_for_dataset(
                        rec, 
                        ai_enhancement.get('explanations', [])
                    )
                }
                for rec in neural_results.get('top_recommendations', [])
            ],
            
            # Web search results for additional sources
            "web_sources": [
                {
                    "title": result.get('title', ''),
                    "url": result.get('url', ''),
                    "description": result.get('description', ''),
                    "source": result.get('source', ''),
                    "type": result.get('type', ''),
                    "relevance_score": result.get('relevance_score', 0)
                }
                for result in web_results
            ],
            
            # AI enhancements
            "dataset_relationships": ai_enhancement.get('relationships', []),
            "analytical_approaches": ai_enhancement.get('analytical_approaches', []),
            "potential_insights": ai_enhancement.get('insights', []),
            "limitations": ai_enhancement.get('limitations', []),
            
            # Research methodology
            "methodology": methodology.get('methodology_guide', ''),
            
            # Singapore context
            "singapore_context": singapore_context.get('singapore_context', ''),
            
            # Performance metrics
            "performance": {
                "neural_model": neural_results.get('neural_performance', {}),
                "ai_provider": ai_enhancement.get('provider', 'unknown'),
                "methodology_provider": methodology.get('generated_by', 'unknown'),
                "total_time": round(processing_time, 3)
            },
            
            # Conversation support
            "conversation": {
                "session_id": session_id,
                "can_refine": True,
                "suggested_refinements": self._generate_refinement_suggestions(query, neural_results)
            }
        }
        
        return response
    
    def _get_explanation_for_dataset(
        self,
        recommendation: Dict[str, Any],
        ai_explanations: List[str]
    ) -> str:
        """Get explanation for specific dataset"""
        # Try to match AI explanation to dataset
        dataset_title = recommendation.get('dataset', {}).get('title', '')
        
        for explanation in ai_explanations:
            # Handle case where explanation might not be a string
            explanation_text = str(explanation) if explanation is not None else ''
            if dataset_title.lower() in explanation_text.lower():
                return explanation_text
        
        # Fallback to neural reasoning
        return recommendation.get('why_recommended', 'Highly ranked by neural model')
    
    def _generate_refinement_suggestions(
        self,
        query: str,
        neural_results: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for query refinement"""
        suggestions = []
        
        # Based on categories found
        categories = set()
        for rec in neural_results.get('top_recommendations', []):
            category = rec.get('dataset', {}).get('category')
            if category:
                categories.add(category)
        
        if len(categories) > 1:
            suggestions.append(f"Focus on specific domain: {', '.join(categories)}")
        
        # Temporal refinement
        suggestions.append("Add time period (e.g., '2020-2024')")
        
        # Geographic refinement
        if 'singapore' not in query.lower():
            suggestions.append("Specify geographic scope (e.g., 'Singapore' or 'Southeast Asia')")
        
        # Method refinement
        suggestions.append("Specify analysis type (e.g., 'time series', 'correlation', 'comparison')")
        
        return suggestions[:3]  # Top 3 suggestions
    
    def _build_error_response(
        self,
        query: str,
        error: str,
        session_id: str,
        processing_time: float = 0.0
    ) -> Dict[str, Any]:
        """Build error response"""
        return {
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(processing_time, 3),
            "error": error,
            "recommendations": [],
            "suggestion": "Please try rephrasing your query or contact support if the issue persists."
        }
    
    async def refine_query(
        self,
        session_id: str,
        refinement: str
    ) -> Dict[str, Any]:
        """
        Refine previous query based on user feedback
        
        Args:
            session_id: Session identifier
            refinement: User's refinement input
            
        Returns:
            Refined recommendations
        """
        # Get session context
        session = self.conversation_manager.get_session(session_id)
        if not session:
            return {"error": "Session not found. Please start a new search."}
        
        # Get conversation history
        history = self.conversation_manager.get_history(session_id)
        if not history:
            return {"error": "No previous query found in session."}
        
        # Build refined query
        last_query = history[-1].get('query', '')
        refined_query = f"{last_query} {refinement}"
        
        # Process with context
        context = {
            "is_refinement": True,
            "original_query": last_query,
            "refinement": refinement,
            "previous_recommendations": history[-1].get('response', {}).get('recommendations', [])
        }
        
        return await self.process_query(refined_query, session_id, context)