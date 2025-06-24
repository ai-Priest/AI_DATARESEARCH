"""
AI-Powered Dataset Research Assistant
High-performance neural recommendations with intelligent enhancement
"""

from .research_assistant import ResearchAssistant
from .llm_clients import LLMManager
from .neural_ai_bridge import NeuralAIBridge
from .conversation_manager import ConversationManager
from .evaluation_metrics import EvaluationMetrics
from .ai_config_manager import AIConfigManager

__version__ = "1.0.0"
__all__ = [
    "ResearchAssistant",
    "LLMManager", 
    "NeuralAIBridge",
    "ConversationManager",
    "EvaluationMetrics",
    "AIConfigManager"
]

# Module metadata
__author__ = "AI Research Assistant Team"
__description__ = "Transforms 69.99% NDCG@3 neural performance into exceptional user experience"