"""
Optimized Research Assistant with Parallel Processing
Implements response time optimization and parallel LLM calls for sub-10s performance
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import as_completed
import json
from datetime import datetime

from .ai_config_manager import AIConfigManager
from .llm_clients import LLMManager
from .neural_ai_bridge import NeuralAIBridge
from .conversation_manager import ConversationManager
from .evaluation_metrics import EvaluationMetrics

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
        self.conversation_manager = ConversationManager(config)
        self.evaluation_metrics = EvaluationMetrics(config)
        
        # Performance configuration
        self.response_config = self.ai_pipeline_config.get('response_settings', {})
        self.max_response_time = self.response_config.get('max_response_time', 8.0)  # Reduced target
        self.neural_timeout = self.response_config.get('neural_inference_timeout', 0.5)
        self.llm_timeout = self.response_config.get('llm_enhancement_timeout', 7.0)
        
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
            
            # Phase 1: Neural inference (fast, parallel with LLM prep)
            neural_task = asyncio.create_task(self._get_neural_recommendations(query))
            
            # Phase 2: Prepare LLM tasks (parallel execution)
            llm_tasks = self._prepare_llm_tasks(query)
            
            # Phase 3: Execute neural + LLM in parallel
            results = await self._execute_parallel_processing(neural_task, llm_tasks, start_time)
            
            # Phase 4: Combine and finalize response
            response = await self._finalize_response(
                query=query,
                session_id=session_id,
                neural_results=results['neural'],
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
    
    def _prepare_llm_tasks(self, query: str) -> List[asyncio.Task]:
        """Prepare LLM tasks for parallel execution."""
        tasks = []
        
        # Task 1: Generate explanations (high priority)
        if self.response_config.get('include_explanations', True):
            explanation_prompt = self._create_explanation_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('explanation', explanation_prompt, timeout=3.0)
            )
            tasks.append(task)
        
        # Task 2: Generate methodology (medium priority)
        if self.response_config.get('include_methodology', True):
            methodology_prompt = self._create_methodology_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('methodology', methodology_prompt, timeout=4.0)
            )
            tasks.append(task)
        
        # Task 3: Singapore context (low priority)
        if self.response_config.get('include_singapore_context', True):
            context_prompt = self._create_context_prompt(query)
            task = asyncio.create_task(
                self._llm_call_with_timeout('context', context_prompt, timeout=5.0)
            )
            tasks.append(task)
        
        return tasks
    
    async def _execute_parallel_processing(self, 
                                         neural_task: asyncio.Task,
                                         llm_tasks: List[asyncio.Task],
                                         start_time: float) -> Dict[str, Any]:
        """Execute neural and LLM tasks in parallel with time management."""
        results = {'neural': None, 'llm': {}}
        
        # Execute all tasks concurrently
        all_tasks = [neural_task] + llm_tasks
        
        try:
            # Wait for tasks with overall timeout
            remaining_time = self.max_response_time - (time.time() - start_time)
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=max(0.5, remaining_time - 0.5)  # Leave 0.5s buffer
            )
            
            # Process results
            results['neural'] = completed_tasks[0] if not isinstance(completed_tasks[0], Exception) else None
            
            for i, task_result in enumerate(completed_tasks[1:]):
                if not isinstance(task_result, Exception) and task_result:
                    task_type = ['explanation', 'methodology', 'context'][i]
                    results['llm'][task_type] = task_result
            
        except asyncio.TimeoutError:
            logger.warning("Parallel processing timeout - using partial results")
            
            # Get any completed results
            if neural_task.done() and not neural_task.exception():
                try:
                    results['neural'] = neural_task.result()
                except Exception as e:
                    logger.warning(f"Neural task error: {e}")
                
            for i, task in enumerate(llm_tasks):
                if task.done() and not task.exception():
                    try:
                        task_type = ['explanation', 'methodology', 'context'][i]
                        results['llm'][task_type] = task.result()
                    except Exception as e:
                        logger.warning(f"LLM task {i} error: {e}")
        
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
                
                recommendations.append({
                    'dataset': rec,
                    'confidence': rec.get('confidence', 0.5),
                    'explanation': explanation,
                    'source': 'neural_optimized',
                    'methodology': llm_results.get('methodology', {}).get('content', 'Neural ranking analysis')
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
            'singapore_context': llm_results.get('context', {}).get('content', 'Singapore government data prioritized')
        }
        
        # Add to conversation history
        response_summary = {
            'recommendations': recommendations,
            'processing_time': processing_time,
            'performance': performance
        }
        
        self.conversation_manager.add_to_history(session_id, query, response_summary)
        
        # Final response
        return {
            'session_id': session_id,
            'query': query,
            'recommendations': recommendations,
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
        return {
            'session_id': session_id,
            'query': query,
            'recommendations': [],
            'error': 'Processing timeout or error',
            'fallback': True,
            'processing_time': processing_time,
            'performance': {
                'processing_time': processing_time,
                'target_time': self.max_response_time,
                'achieved_target': False,
                'error': error
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