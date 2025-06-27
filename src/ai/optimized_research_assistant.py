"""
Optimized Research Assistant with Parallel Processing
Implements response time optimization and parallel LLM calls for sub-10s performance
"""

import asyncio
import json
import logging
import time
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from .ai_config_manager import AIConfigManager
from .conversation_manager import ConversationManager
from .evaluation_metrics import EvaluationMetrics
from .llm_clients import LLMManager
from .neural_ai_bridge import NeuralAIBridge
from .web_search_engine import WebSearchEngine
from .url_validator import url_validator

logger = logging.getLogger(__name__)


class OptimizedResearchAssistant:
    """
    Optimized research assistant with parallel processing and sub-10s response times.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ai_pipeline_config = config.get('ai_pipeline', {})
        
        # Initialize components
        self.llm_manager = LLMManager(config)
        self.neural_bridge = NeuralAIBridge(config)
        self.web_search_engine = WebSearchEngine(config)
        self.conversation_manager = ConversationManager(config)
        self.evaluation_metrics = EvaluationMetrics(config)
        
        # Performance configuration
        self.response_config = self.ai_pipeline_config.get('response_settings', {})
        self.max_response_time = self.response_config.get('max_response_time', 5.0)  # Further reduced
        self.neural_timeout = self.response_config.get('neural_inference_timeout', 0.3)
        self.llm_timeout = self.response_config.get('llm_enhancement_timeout', 4.0)  # Much shorter
        
        # Parallel processing settings
        self.enable_parallel_processing = True
        self.max_concurrent_llm_calls = 3
        
        logger.info(f"🚀 OptimizedResearchAssistant initialized with {self.max_response_time}s target")
    
    async def process_query_optimized(self, 
                                    query: str, 
                                    session_id: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Process query with optimized parallel execution for sub-10s response.
        
        Args:
            query: User search query
            session_id: Optional session ID for conversation tracking
            **kwargs: Additional parameters
            
        Returns:
            Optimized response dictionary with performance metrics
        """
        start_time = time.time()
        logger.info(f"🔍 Processing optimized query: '{query[:50]}...'")
        
        try:
            # Create session if needed
            if not session_id:
                session = self.conversation_manager.create_session()
                session_id = session['session_id']
            
            # Phase 1: Neural inference + Web search (parallel execution)
            neural_task = asyncio.create_task(self._get_neural_recommendations(query))
            web_search_task = asyncio.create_task(self._get_web_search_results(query))
            
            # Phase 2: Prepare LLM tasks (parallel execution)
            llm_tasks = self._prepare_llm_tasks(query)
            
            # Phase 3: Execute neural + web search + LLM in parallel
            results = await self._execute_parallel_processing(neural_task, web_search_task, llm_tasks, start_time)
            
            # Phase 4: Combine and finalize response
            response = await self._finalize_response(
                query=query,
                session_id=session_id,
                neural_results=results['neural'],
                web_search_results=results['web_search'],
                llm_results=results['llm'],
                start_time=start_time,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Query processed in {processing_time:.2f}s (target: {self.max_response_time}s)")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query processing failed in {processing_time:.2f}s: {str(e)}")
            
            # Return fallback response
            return await self._create_fallback_response(query, session_id, str(e), processing_time)
    
    async def _get_neural_recommendations(self, query: str) -> Dict[str, Any]:
        """Get neural recommendations with timeout."""
        try:
            neural_results = await asyncio.wait_for(
                self.neural_bridge.get_neural_recommendations(query, top_k=5),
                timeout=self.neural_timeout
            )
            return neural_results
        except asyncio.TimeoutError:
            logger.warning(f"Neural inference timeout ({self.neural_timeout}s)")
            return self._create_fallback_neural_response(query)
        except Exception as e:
            logger.warning(f"Neural inference error: {e}")
            return self._create_fallback_neural_response(query)
    
    async def _get_web_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Get web search results with timeout."""
        try:
            web_timeout = self.config.get('web_search', {}).get('timeout', 10)
            web_results = await asyncio.wait_for(
                self.web_search_engine.search_web(query),
                timeout=web_timeout
            )
            return web_results
        except asyncio.TimeoutError:
            logger.warning(f"Web search timeout ({web_timeout}s)")
            return []
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []
    
    def _prepare_llm_tasks(self, query: str) -> List[asyncio.Task]:
        """Prepare LLM tasks for parallel execution."""
        tasks = []
        
        # Task 1: Generate explanations (high priority) - much shorter timeout
        if self.response_config.get('include_explanations', True):
            explanation_prompt = self._create_explanation_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('explanation', explanation_prompt, timeout=8.0)
            )
            tasks.append(task)
        
        # Task 2: Generate methodology (medium priority) - shorter timeout
        if self.response_config.get('include_methodology', True):
            methodology_prompt = self._create_methodology_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('methodology', methodology_prompt, timeout=10.0)
            )
            tasks.append(task)
        
        # Task 3: Singapore context (low priority) - shorter timeout
        if self.response_config.get('include_singapore_context', True):
            context_prompt = self._create_context_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('context', context_prompt, timeout=12.0)
            )
            tasks.append(task)
        
        return tasks
    
    async def _execute_parallel_processing(self, 
                                         neural_task: asyncio.Task,
                                         web_search_task: asyncio.Task,
                                         llm_tasks: List[asyncio.Task],
                                         start_time: float) -> Dict[str, Any]:
        """Execute neural, web search, and LLM tasks in parallel with time management."""
        results = {'neural': None, 'web_search': [], 'llm': {}}
        
        # Execute all tasks concurrently
        all_tasks = [neural_task, web_search_task] + llm_tasks
        
        try:
            # Wait for tasks with overall timeout
            remaining_time = self.max_response_time - (time.time() - start_time)
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=max(0.5, remaining_time - 0.5)  # Leave 0.5s buffer
            )
            
            # Process results
            results['neural'] = completed_tasks[0] if not isinstance(completed_tasks[0], Exception) else None
            results['web_search'] = completed_tasks[1] if not isinstance(completed_tasks[1], Exception) else []
            
            for i, task_result in enumerate(completed_tasks[2:]):
                if not isinstance(task_result, Exception) and task_result:
                    task_type = ['explanation', 'methodology', 'context'][i]
                    results['llm'][task_type] = task_result
            
        except asyncio.TimeoutError:
            logger.warning("Parallel processing timeout - using partial results")
            
            # Cancel any remaining tasks to prevent CancelledError
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait a bit for cancellations to complete
            try:
                await asyncio.wait_for(asyncio.gather(*all_tasks, return_exceptions=True), timeout=0.1)
            except:
                pass  # Ignore cancellation errors
            
            # Get any completed results safely
            try:
                if neural_task.done() and not neural_task.cancelled() and not neural_task.exception():
                    results['neural'] = neural_task.result()
            except Exception as e:
                logger.warning(f"Neural task error: {e}")
                
            for i, task in enumerate(llm_tasks):
                try:
                    if task.done() and not task.cancelled() and not task.exception():
                        task_type = ['explanation', 'methodology', 'context'][i]
                        results['llm'][task_type] = task.result()
                except Exception as e:
                    logger.warning(f"LLM task {i} error: {e}")
        
        except Exception as e:
            logger.error(f"Parallel processing critical error: {e}")
            # Ensure we have at least neural results
            try:
                if not results.get('neural') and neural_task.done() and not neural_task.cancelled():
                    results['neural'] = neural_task.result()
            except Exception:
                pass
        
        return results
    
    async def _llm_call_with_timeout(self, 
                                   task_type: str, 
                                   prompt: str, 
                                   timeout: float) -> Optional[Dict[str, Any]]:
        """Make LLM call with timeout and error handling."""
        try:
            result = await asyncio.wait_for(
                self.llm_manager.complete_with_fallback(
                    prompt=prompt,
                    max_tokens=256,  # Reduced for speed
                    temperature=0.6
                ),
                timeout=timeout
            )
            logger.debug(f"✅ {task_type} completed in <{timeout}s")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ {task_type} timeout ({timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"❌ {task_type} error: {e}")
            return None
    
    async def _finalize_response(self,
                               query: str,
                               session_id: str,
                               neural_results: Optional[Dict],
                               web_search_results: List[Dict[str, Any]],
                               llm_results: Dict[str, Any],
                               start_time: float,
                               **kwargs) -> Dict[str, Any]:
        """Finalize and structure the response."""
        
        # Extract neural recommendations
        recommendations = []
        if neural_results and 'recommendations' in neural_results:
            for rec in neural_results['recommendations']:
                # Add LLM-generated explanation if available
                explanation = self._get_explanation_for_dataset(
                    rec, llm_results.get('explanation')
                )
                
                # Validate and correct dataset URL
                validated_dataset = await url_validator.validate_and_correct_dataset(rec)
                
                recommendations.append({
                    'dataset': validated_dataset,
                    'confidence': rec.get('confidence', 0.5),
                    'explanation': explanation,
                    'source': 'neural_optimized',
                    'methodology': llm_results.get('methodology', {}).get('content', 'Neural ranking analysis') if llm_results.get('methodology') else 'Neural ranking analysis'
                })
        
        # Create performance metrics
        processing_time = time.time() - start_time
        performance = {
            'processing_time': processing_time,
            'target_time': self.max_response_time,
            'performance_ratio': processing_time / self.max_response_time,
            'neural_inference_time': neural_results.get('neural_metrics', {}).get('inference_time', 0.0) if neural_results else 0.0,
            'ai_provider': self._determine_primary_provider(llm_results),
            'llm_tasks_completed': len([k for k, v in llm_results.items() if v]),
            'optimization_level': 'high'
        }
        
        # Create conversation context
        conversation = {
            'session_id': session_id,
            'can_refine': True,
            'suggested_refinements': self._generate_quick_refinements(query),
            'singapore_context': llm_results.get('context', {}).get('content', 'Singapore government data prioritized') if llm_results.get('context') else 'Singapore government data prioritized'
        }
        
        # Add to conversation history
        response_summary = {
            'recommendations': recommendations,
            'processing_time': processing_time,
            'performance': performance
        }
        
        self.conversation_manager.add_to_history(session_id, query, response_summary)
        
        # Merge all results into a unified list
        all_results = []
        
        # Add local recommendations with normalized structure
        for rec in recommendations:
            dataset = rec.get('dataset', {})
            all_results.append({
                'title': dataset.get('title', ''),
                'url': dataset.get('url', ''),
                'description': dataset.get('description', ''),
                'source': dataset.get('source', 'local'),
                'type': 'local_dataset',
                'relevance_score': dataset.get('relevance_score', 0) * 1000,  # Boost local scores
                'confidence': rec.get('confidence', 0.5),
                'explanation': rec.get('explanation', ''),
                'dataset_info': dataset  # Keep full dataset info
            })
        
        # Add web sources with existing structure
        for result in web_search_results:
            all_results.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'description': result.get('description', ''),
                'source': result.get('source', ''),
                'type': result.get('type', 'web_source'),
                'relevance_score': result.get('relevance_score', 0),
                'confidence': 0.7,  # Default confidence for web sources
                'explanation': '',
                'dataset_info': result
            })
        
        # Sort all results by relevance score (highest first)
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Final response with unified results
        return {
            'session_id': session_id,
            'query': query,
            'recommendations': recommendations,  # Keep for backward compatibility
            'web_sources': [  # Keep for backward compatibility
                {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'description': result.get('description', ''),
                    'source': result.get('source', ''),
                    'type': result.get('type', ''),
                    'relevance_score': result.get('relevance_score', 0)
                }
                for result in web_search_results
            ],
            'all_results': all_results[:20],  # New unified list (limit to top 20)
            'conversation': conversation,
            'performance': performance,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'optimization': {
                'parallel_processing': True,
                'neural_timeout': self.neural_timeout,
                'llm_timeout': sum([3.0, 4.0, 5.0]),  # Total LLM time budget
                'achieved_target': processing_time <= self.max_response_time
            }
        }
    
    def _create_explanation_prompt(self, query: str) -> str:
        """Create focused prompt for dataset explanations."""
        return f"""Given the query '{query}', provide a brief explanation (max 100 words) of why recommended datasets are relevant. Focus on:
1. Direct relevance to query
2. Data quality and reliability
3. Practical research value"""
    
    def _create_methodology_prompt(self, query: str) -> str:
        """Create focused prompt for research methodology."""
        return f"""For research query '{query}', suggest a concise methodology (max 80 words):
1. Data integration approach
2. Analysis methods
3. Validation steps"""
    
    def _create_context_prompt(self, query: str) -> str:
        """Create focused prompt for Singapore context."""
        return f"""For '{query}' in Singapore context, provide brief insights (max 60 words):
1. Local relevance
2. Government data sources
3. Regional considerations"""
    
    def _get_explanation_for_dataset(self, 
                                   dataset: Dict, 
                                   explanation_result: Optional[Dict]) -> str:
        """Generate explanation for specific dataset."""
        if explanation_result and 'content' in explanation_result:
            # Use LLM-generated explanation
            return explanation_result['content'][:200] + "..."
        else:
            # Fallback to rule-based explanation
            title = dataset.get('title', 'Dataset')
            confidence = dataset.get('confidence', 0.5)
            return f"{title} is recommended with {confidence*100:.0f}% confidence based on neural ranking analysis."
    
    def _determine_primary_provider(self, llm_results: Dict[str, Any]) -> str:
        """Determine which LLM provider was primarily used."""
        providers = []
        for result in llm_results.values():
            if result and 'provider' in result:
                providers.append(result['provider'])
        
        if providers:
            # Return most common provider
            return max(set(providers), key=providers.count)
        return 'neural_only'
    
    def _generate_quick_refinements(self, query: str) -> List[str]:
        """Generate quick refinement suggestions."""
        refinements = [
            "focus on recent data from 2024",
            "include government sources only",
            "add geographic breakdown",
            "include historical trends"
        ]
        return refinements[:2]  # Return top 2 for speed
    
    async def _create_fallback_response(self, 
                                      query: str, 
                                      session_id: str, 
                                      error: str, 
                                      processing_time: float) -> Dict[str, Any]:
        """Create fallback response when main processing fails."""
        try:
            # Try to get neural recommendations as fallback
            neural_results = await self.neural_bridge.get_neural_recommendations(query, top_k=5)
            datasets = neural_results.get('recommendations', [])
            
            return {
                'session_id': session_id,
                'query': query,
                'response': f"Found {len(datasets)} datasets for '{query}'. AI enhancement unavailable due to timeout, showing neural search results.",
                'recommendations': datasets,  # Changed from 'datasets' to 'recommendations'
                'web_sources': [],  # Empty web sources since fallback
                'fallback': True,
                'processing_time': processing_time,
                'metadata': {
                    'from_cache': False,
                    'fallback_mode': True,
                    'ai_enhancement': 'unavailable'
                },
                'performance': {
                    'processing_time': processing_time,
                    'target_time': self.max_response_time,
                    'achieved_target': False,
                    'error': error
                }
            }
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return {
                'session_id': session_id,
                'query': query,
                'response': "Search temporarily unavailable. Please try the standard search mode.",
                'recommendations': [],  # Changed from 'datasets' to 'recommendations'
                'web_sources': [],  # Empty web sources since fallback
                'fallback': True,
                'processing_time': processing_time,
                'metadata': {
                    'from_cache': False,
                    'fallback_mode': True,
                    'ai_enhancement': 'unavailable'
                },
                'performance': {
                    'processing_time': processing_time,
                    'target_time': self.max_response_time,
                    'achieved_target': False,
                    'error': f"Both main and fallback failed: {error}, {fallback_error}"
                }
            }
    
    def _create_fallback_neural_response(self, query: str) -> Dict[str, Any]:
        """Create fallback neural response."""
        return {
            'recommendations': [],
            'neural_metrics': {
                'model': 'fallback',
                'inference_time': 0.0,
                'ndcg_at_3': 0.0
            },
            'fallback': True
        }
    
    async def refine_query_optimized(self, 
                                   session_id: str, 
                                   refinement: str) -> Dict[str, Any]:
        """Optimized query refinement with faster processing."""
        session_data = self.conversation_manager.get_session_summary(session_id)
        
        if not session_data or not session_data.get('history'):
            return {'error': 'No session history found'}
        
        # Get last query
        last_entry = session_data['history'][-1]
        original_query = last_entry.get('query', '')
        
        # Create refined query
        refined_query = f"{original_query} {refinement}"
        
        # Process with optimization
        return await self.process_query_optimized(refined_query, session_id)


def create_optimized_research_assistant(config_path: str = "config/ai_config.yml") -> OptimizedResearchAssistant:
    """Create optimized research assistant instance."""
    config_manager = AIConfigManager(config_path)
    return OptimizedResearchAssistant(config_manager.config)