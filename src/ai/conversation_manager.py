"""
Conversation Manager for session handling and history tracking
Enables multi-turn conversations and context preservation
"""
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation sessions and history for the research assistant
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_config = config.get('session_config', {})
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = self.session_config.get('session_timeout', 3600)  # 1 hour default
        self.max_history_length = self.session_config.get('max_history_length', 20)
        self.storage_backend = self.session_config.get('storage_backend', 'memory')
        
        # Initialize storage
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialize storage backend"""
        if self.storage_backend == 'memory':
            # In-memory storage (default)
            logger.info("Using in-memory session storage")
        elif self.storage_backend == 'redis':
            # Redis storage for production
            # Would implement Redis connection here
            logger.info("Redis storage not implemented, falling back to memory")
            self.storage_backend = 'memory'
    
    def create_session(self) -> Dict[str, Any]:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        session = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "history": [],
            "context": {},
            "preferences": {},
            "metadata": {
                "query_count": 0,
                "refinement_count": 0,
                "total_datasets_shown": 0
            }
        }
        
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an existing session"""
        session = self.sessions.get(session_id)
        
        if session:
            # Check if session expired
            if self._is_session_expired(session):
                logger.info(f"Session {session_id} expired")
                self.end_session(session_id)
                return None
            
            # Update last active time
            session['last_active'] = time.time()
            return session
        
        return None
    
    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Check if session has expired"""
        last_active = session.get('last_active', 0)
        return (time.time() - last_active) > self.session_timeout
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """Add a message to session history for conversation tracking"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Add message to conversation history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        session['conversation_history'].append(message)
        
        # Keep only recent messages (last 20)
        if len(session['conversation_history']) > 20:
            session['conversation_history'] = session['conversation_history'][-20:]
        
        session['last_active'] = time.time()
        logger.info(f"Added {role} message to session {session_id}")
        return True
    
    def add_to_history(
        self,
        session_id: str,
        query: str,
        response: Dict[str, Any]
    ) -> bool:
        """Add query and response to session history"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Create history entry
        history_entry = {
            "timestamp": time.time(),
            "query": query,
            "response": self._compress_response(response),
            "interaction_type": "refinement" if response.get('conversation', {}).get('is_refinement') else "query"
        }
        
        # Add to history
        session['history'].append(history_entry)
        
        # Trim history if too long
        if len(session['history']) > self.max_history_length:
            session['history'] = session['history'][-self.max_history_length:]
        
        # Update metadata
        session['metadata']['query_count'] += 1
        if history_entry['interaction_type'] == 'refinement':
            session['metadata']['refinement_count'] += 1
        
        # Count total datasets shown
        dataset_count = len(response.get('recommendations', []))
        session['metadata']['total_datasets_shown'] += dataset_count
        
        logger.info(f"Added to history for session {session_id}: {query[:50]}...")
        return True
    
    def _compress_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compress response for storage in history"""
        compressed = {
            "recommendations": [
                {
                    "dataset_id": rec.get('dataset', {}).get('id'),
                    "title": rec.get('dataset', {}).get('title'),
                    "confidence": rec.get('confidence', 0)
                }
                for rec in response.get('recommendations', [])[:3]  # Keep top 3
            ],
            "processing_time": response.get('processing_time'),
            "ai_provider": response.get('performance', {}).get('ai_provider')
        }
        
        return compressed
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return session.get('history', [])
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Build context from history
        context = {
            "session_id": session_id,
            "query_count": session['metadata']['query_count'],
            "previous_queries": [h['query'] for h in session['history'][-3:]],  # Last 3 queries
            "previous_datasets": self._extract_previous_datasets(session['history']),
            "preferences": session.get('preferences', {}),
            "session_duration": time.time() - session['created_at']
        }
        
        return context
    
    def _extract_previous_datasets(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract unique datasets from history"""
        datasets = set()
        
        for entry in history[-5:]:  # Last 5 interactions
            for rec in entry.get('response', {}).get('recommendations', []):
                title = rec.get('title')
                if title:
                    datasets.add(title)
        
        return list(datasets)
    
    def update_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences for a session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['preferences'].update(preferences)
        logger.info(f"Updated preferences for session {session_id}")
        return True
    
    def end_session(self, session_id: str) -> bool:
        """End a conversation session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Log session summary
            logger.info(f"Ending session {session_id}: "
                       f"{session['metadata']['query_count']} queries, "
                       f"{session['metadata']['refinement_count']} refinements, "
                       f"{session['metadata']['total_datasets_shown']} datasets shown")
            
            # Remove from active sessions
            del self.sessions[session_id]
            return True
        
        return False
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session"""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        duration = time.time() - session['created_at']
        
        summary = {
            "session_id": session_id,
            "created_at": datetime.fromtimestamp(session['created_at']).isoformat(),
            "duration_seconds": round(duration),
            "duration_formatted": str(timedelta(seconds=int(duration))),
            "statistics": {
                "total_queries": session['metadata']['query_count'],
                "refinements": session['metadata']['refinement_count'],
                "datasets_shown": session['metadata']['total_datasets_shown'],
                "avg_datasets_per_query": round(
                    session['metadata']['total_datasets_shown'] / max(session['metadata']['query_count'], 1),
                    1
                )
            },
            "queries": [h['query'] for h in session['history']],
            "is_active": True
        }
        
        return summary
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = []
        current_time = time.time()
        
        for session_id, session in self.sessions.items():
            if (current_time - session['last_active']) > self.session_timeout:
                expired.append(session_id)
        
        for session_id in expired:
            self.end_session(session_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        self.cleanup_expired_sessions()
        return len(self.sessions)
    
    def export_session(self, session_id: str) -> Optional[str]:
        """Export session data as JSON"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Prepare export data
        export_data = {
            "session_id": session_id,
            "created_at": datetime.fromtimestamp(session['created_at']).isoformat(),
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_session_summary(session_id),
            "full_history": [
                {
                    "timestamp": datetime.fromtimestamp(h['timestamp']).isoformat(),
                    "query": h['query'],
                    "recommendations_count": len(h.get('response', {}).get('recommendations', [])),
                    "top_recommendations": h.get('response', {}).get('recommendations', [])
                }
                for h in session['history']
            ]
        }
        
        return json.dumps(export_data, indent=2)