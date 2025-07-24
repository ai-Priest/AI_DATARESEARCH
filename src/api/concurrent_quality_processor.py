"""
Concurrent Quality Processor
Implements parallel processing of neural, web, and LLM components
while maintaining quality standards under load
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import PriorityQueue
from threading import Lock, Semaphore
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class QualityRequest:
    """Request with quality requirements and priority"""
    request_id: str
    query: str
    quality_threshold: float
    priority: int  # Lower number = higher priority
    timestamp: float
    timeout_seconds: float
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """Priority queue comparison"""
        return self.priority < other.priority


@dataclass
class ProcessingResult:
    """Result from concurrent processing with quality metrics"""
    request_id: str
    component: str  # 'neural', 'web', 'llm'
    success: bool
    data: Any
    quality_score: float
    processing_time: float
    error: Optional[str] = None


class QualityAwareRequestQueue:
    """Request queue that maintains quality standards under load"""
    
    def __init__(self, max_concurrent_requests: int = 10, quality_threshold: float = 0.7):
        self.max_concurrent_requests = max_concurrent_requests
        self.quality_threshold = quality_threshold
        
        # Queue management
        self.request_queue = PriorityQueue()
        self.active_requests: Dict[str, QualityRequest] = {}
        self.completed_requests: Dict[str, List[ProcessingResult]] = {}
        
        # Concurrency control
        self.semaphore = Semaphore(max_concurrent_requests)
        self.lock = Lock()
        
        # Quality tracking
        self.quality_stats = {
            'total_requests': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'avg_processing_time': 0.0,
            'concurrent_peak': 0
        }
        
        logger.info(f"🚦 QualityAwareRequestQueue initialized: max_concurrent={max_concurrent_requests}")
    
    def submit_request(self, request: QualityRequest) -> str:
        """Submit request to quality-aware queue"""
        with self.lock:
            # Check if system is overloaded
            if len(self.active_requests) >= self.max_concurrent_requests * 2:
                # Reject low-priority requests when overloaded
                if request.priority > 5:
                    raise Exception("System overloaded - rejecting low priority request")
            
            # Add to queue
            self.request_queue.put(request)
            self.quality_stats['total_requests'] += 1
            
            logger.debug(f"📥 Request queued: {request.request_id} (priority: {request.priority})")
            
            return request.request_id
    
    async def process_queue(self):
        """Process requests from queue with quality maintenance"""
        while True:
            try:
                # Get next request (non-blocking)
                if not self.request_queue.empty():
                    try:
                        request = self.request_queue.get_nowait()
                        
                        # Check if request has timed out
                        if time.time() - request.timestamp > request.timeout_seconds:
                            logger.warning(f"⏰ Request timed out: {request.request_id}")
                            continue
                        
                        # Process request with concurrency control (fire and forget)
                        asyncio.create_task(self._process_request_with_quality_control(request))
                        
                    except Exception as e:
                        # Queue is empty or other error
                        await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_request_with_quality_control(self, request: QualityRequest):
        """Process request with quality control and concurrency limits"""
        
        # Acquire semaphore for concurrency control
        self.semaphore.acquire()
        
        try:
            with self.lock:
                self.active_requests[request.request_id] = request
                current_concurrent = len(self.active_requests)
                self.quality_stats['concurrent_peak'] = max(
                    self.quality_stats['concurrent_peak'], current_concurrent
                )
            
            # Process request
            start_time = time.time()
            results = await self._process_request_components(request)
            processing_time = time.time() - start_time
            
            # Validate quality
            quality_passed = self._validate_results_quality(results, request.quality_threshold)
            
            # Update statistics
            with self.lock:
                if quality_passed:
                    self.quality_stats['quality_passed'] += 1
                else:
                    self.quality_stats['quality_failed'] += 1
                
                # Update average processing time
                total = self.quality_stats['total_requests']
                current_avg = self.quality_stats['avg_processing_time']
                self.quality_stats['avg_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
            
            # Store results
            with self.lock:
                self.completed_requests[request.request_id] = results
                # Keep request info for quality validation
                # Don't delete from active_requests immediately
            
            # Call callback if provided
            if request.callback:
                try:
                    await request.callback(request.request_id, results, quality_passed)
                except Exception as e:
                    logger.error(f"Callback error for {request.request_id}: {e}")
            
            logger.debug(f"✅ Request completed: {request.request_id} (quality: {quality_passed})")
            
        finally:
            self.semaphore.release()
    
    async def _process_request_components(self, request: QualityRequest) -> List[ProcessingResult]:
        """Process request components concurrently"""
        # This would integrate with actual neural, web, and LLM components
        # For now, we'll simulate the processing
        
        tasks = [
            self._process_neural_component(request),
            self._process_web_component(request),
            self._process_llm_component(request)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_names = ['neural', 'web', 'llm']
                processed_results.append(ProcessingResult(
                    request_id=request.request_id,
                    component=component_names[i],
                    success=False,
                    data=None,
                    quality_score=0.0,
                    processing_time=0.0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_neural_component(self, request: QualityRequest) -> ProcessingResult:
        """Process neural component with quality validation"""
        start_time = time.time()
        
        try:
            # Simulate neural processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate quality-aware neural results
            quality_score = 0.8 if 'psychology' in request.query.lower() else 0.7
            
            data = {
                'recommendations': [
                    {'source': 'Neural Source 1', 'score': quality_score},
                    {'source': 'Neural Source 2', 'score': quality_score - 0.1}
                ]
            }
            
            return ProcessingResult(
                request_id=request.request_id,
                component='neural',
                success=True,
                data=data,
                quality_score=quality_score,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                request_id=request.request_id,
                component='neural',
                success=False,
                data=None,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _process_web_component(self, request: QualityRequest) -> ProcessingResult:
        """Process web search component with quality validation"""
        start_time = time.time()
        
        try:
            # Simulate web search processing
            await asyncio.sleep(0.05)  # Simulate processing time
            
            # Simulate quality-aware web results
            quality_score = 0.75
            
            data = {
                'web_results': [
                    {'url': 'https://example.com/dataset1', 'relevance': quality_score},
                    {'url': 'https://example.com/dataset2', 'relevance': quality_score - 0.1}
                ]
            }
            
            return ProcessingResult(
                request_id=request.request_id,
                component='web',
                success=True,
                data=data,
                quality_score=quality_score,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                request_id=request.request_id,
                component='web',
                success=False,
                data=None,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _process_llm_component(self, request: QualityRequest) -> ProcessingResult:
        """Process LLM enhancement component with quality validation"""
        start_time = time.time()
        
        try:
            # Simulate LLM processing
            await asyncio.sleep(0.2)  # Simulate processing time
            
            # Simulate quality-aware LLM results
            quality_score = 0.85
            
            data = {
                'enhanced_query': f"Enhanced: {request.query}",
                'explanations': ['This dataset is relevant because...'],
                'quality_assessment': quality_score
            }
            
            return ProcessingResult(
                request_id=request.request_id,
                component='llm',
                success=True,
                data=data,
                quality_score=quality_score,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                request_id=request.request_id,
                component='llm',
                success=False,
                data=None,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _validate_results_quality(self, results: List[ProcessingResult], threshold: float) -> bool:
        """Validate that results meet quality threshold"""
        if not results:
            return False
        
        # Check that at least one component succeeded with good quality
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return False
        
        # Check average quality score
        avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
        
        return avg_quality >= threshold
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a request"""
        with self.lock:
            if request_id in self.active_requests:
                return {
                    'status': 'processing',
                    'request': self.active_requests[request_id]
                }
            elif request_id in self.completed_requests:
                results = self.completed_requests[request_id]
                # Use default threshold if request not found in active requests
                threshold = 0.7
                for req_id, req in self.active_requests.items():
                    if req_id == request_id:
                        threshold = req.quality_threshold
                        break
                
                return {
                    'status': 'completed',
                    'results': results,
                    'quality_passed': self._validate_results_quality(results, threshold)
                }
            else:
                return {'status': 'not_found'}
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue and quality statistics"""
        with self.lock:
            return {
                'queue_size': self.request_queue.qsize(),
                'active_requests': len(self.active_requests),
                'completed_requests': len(self.completed_requests),
                'max_concurrent': self.max_concurrent_requests,
                'quality_stats': self.quality_stats.copy()
            }


class ConcurrentQualityProcessor:
    """Main concurrent processor with quality maintenance"""
    
    def __init__(self, 
                 max_concurrent_requests: int = 10,
                 quality_threshold: float = 0.7,
                 thread_pool_size: int = 4):
        
        self.max_concurrent_requests = max_concurrent_requests
        self.quality_threshold = quality_threshold
        
        # Initialize request queue
        self.request_queue = QualityAwareRequestQueue(
            max_concurrent_requests=max_concurrent_requests,
            quality_threshold=quality_threshold
        )
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Start queue processor
        self.queue_processor_task = None
        
        logger.info(f"🔄 ConcurrentQualityProcessor initialized")
        logger.info(f"  Max concurrent: {max_concurrent_requests}")
        logger.info(f"  Quality threshold: {quality_threshold}")
        logger.info(f"  Thread pool size: {thread_pool_size}")
    
    async def start(self):
        """Start the concurrent processor"""
        if not self.queue_processor_task:
            self.queue_processor_task = asyncio.create_task(
                self.request_queue.process_queue()
            )
            logger.info("🚀 Concurrent quality processor started")
    
    async def stop(self):
        """Stop the concurrent processor"""
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
            self.queue_processor_task = None
        
        self.thread_pool.shutdown(wait=True)
        logger.info("🛑 Concurrent quality processor stopped")
    
    async def process_request(self, 
                            query: str,
                            quality_threshold: Optional[float] = None,
                            priority: int = 5,
                            timeout_seconds: float = 30.0,
                            callback: Optional[Callable] = None) -> str:
        """
        Submit request for concurrent processing with quality maintenance
        
        Args:
            query: Search query
            quality_threshold: Minimum quality threshold (uses default if None)
            priority: Request priority (lower = higher priority)
            timeout_seconds: Request timeout
            callback: Optional callback for completion notification
            
        Returns:
            Request ID for tracking
        """
        
        request = QualityRequest(
            request_id=str(uuid4()),
            query=query,
            quality_threshold=quality_threshold or self.quality_threshold,
            priority=priority,
            timestamp=time.time(),
            timeout_seconds=timeout_seconds,
            callback=callback
        )
        
        request_id = self.request_queue.submit_request(request)
        
        logger.info(f"📤 Request submitted: {request_id} (query: '{query[:50]}...')")
        
        return request_id
    
    async def get_request_results(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a completed request"""
        return self.request_queue.get_request_status(request_id)
    
    async def wait_for_request(self, request_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for request completion with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.request_queue.get_request_status(request_id)
            
            if status['status'] in ['completed', 'not_found']:
                return status
            
            await asyncio.sleep(0.1)
        
        return {'status': 'timeout'}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        queue_stats = self.request_queue.get_queue_statistics()
        
        return {
            'processor': {
                'max_concurrent_requests': self.max_concurrent_requests,
                'quality_threshold': self.quality_threshold,
                'thread_pool_size': self.thread_pool._max_workers
            },
            'queue': queue_stats,
            'system_health': self._assess_system_health(queue_stats)
        }
    
    def _assess_system_health(self, queue_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system health based on statistics"""
        quality_stats = queue_stats['quality_stats']
        
        # Calculate health metrics
        total_requests = quality_stats['total_requests']
        quality_pass_rate = (
            quality_stats['quality_passed'] / total_requests 
            if total_requests > 0 else 1.0
        )
        
        # Determine health status
        if quality_pass_rate >= 0.9 and queue_stats['queue_size'] < 10:
            health_status = 'excellent'
        elif quality_pass_rate >= 0.8 and queue_stats['queue_size'] < 20:
            health_status = 'good'
        elif quality_pass_rate >= 0.7:
            health_status = 'fair'
        else:
            health_status = 'poor'
        
        return {
            'status': health_status,
            'quality_pass_rate': quality_pass_rate,
            'avg_processing_time': quality_stats['avg_processing_time'],
            'concurrent_peak': quality_stats['concurrent_peak'],
            'recommendations': self._get_health_recommendations(health_status, queue_stats)
        }
    
    def _get_health_recommendations(self, health_status: str, queue_stats: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving system health"""
        recommendations = []
        
        if health_status == 'poor':
            recommendations.append("Consider increasing quality threshold")
            recommendations.append("Review and optimize component processing")
        
        if queue_stats['queue_size'] > 20:
            recommendations.append("Consider increasing max concurrent requests")
        
        if queue_stats['quality_stats']['avg_processing_time'] > 5.0:
            recommendations.append("Optimize component processing times")
        
        return recommendations


# Factory function
def create_concurrent_quality_processor(
    max_concurrent_requests: int = 10,
    quality_threshold: float = 0.7,
    thread_pool_size: int = 4
) -> ConcurrentQualityProcessor:
    """Factory function to create concurrent quality processor"""
    
    return ConcurrentQualityProcessor(
        max_concurrent_requests=max_concurrent_requests,
        quality_threshold=quality_threshold,
        thread_pool_size=thread_pool_size
    )


if __name__ == "__main__":
    async def test_concurrent_processor():
        """Test the concurrent quality processor"""
        logging.basicConfig(level=logging.INFO)
        
        processor = create_concurrent_quality_processor(
            max_concurrent_requests=5,
            quality_threshold=0.7
        )
        
        await processor.start()
        
        print("🧪 Testing Concurrent Quality Processor\n")
        
        # Submit multiple requests
        test_queries = [
            "psychology research data",
            "singapore housing statistics", 
            "climate change indicators",
            "machine learning datasets",
            "economic data analysis"
        ]
        
        request_ids = []
        
        # Submit requests with different priorities
        for i, query in enumerate(test_queries):
            request_id = await processor.process_request(
                query=query,
                priority=i + 1,  # Different priorities
                timeout_seconds=10.0
            )
            request_ids.append(request_id)
            print(f"📤 Submitted: {query} (ID: {request_id[:8]}...)")
        
        print(f"\n⏳ Waiting for {len(request_ids)} requests to complete...")
        
        # Wait for all requests to complete
        completed_count = 0
        for request_id in request_ids:
            result = await processor.wait_for_request(request_id, timeout=15.0)
            
            if result['status'] == 'completed':
                completed_count += 1
                quality_passed = result.get('quality_passed', False)
                print(f"✅ Request completed: {request_id[:8]}... (Quality: {'✓' if quality_passed else '✗'})")
            else:
                print(f"❌ Request failed: {request_id[:8]}... (Status: {result['status']})")
        
        # Get system statistics
        stats = processor.get_system_statistics()
        
        print(f"\n📊 System Statistics:")
        print(f"  Completed requests: {completed_count}/{len(request_ids)}")
        print(f"  Quality pass rate: {stats['system_health']['quality_pass_rate']:.3f}")
        print(f"  System health: {stats['system_health']['status']}")
        print(f"  Average processing time: {stats['queue']['quality_stats']['avg_processing_time']:.3f}s")
        print(f"  Concurrent peak: {stats['queue']['quality_stats']['concurrent_peak']}")
        
        await processor.stop()
        
        print("\n✅ Concurrent quality processor testing complete!")
    
    asyncio.run(test_concurrent_processor())