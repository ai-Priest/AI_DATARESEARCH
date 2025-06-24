"""
Optional: vLLM Client for Local Model Inference
Use this ONLY if you want to run models locally on your GPU
Not needed for API-based services (MiniMax, Mistral API, Claude, OpenAI)
"""
import logging
from typing import Dict, List, Optional, Any
from .llm_clients import BaseLLMClient

logger = logging.getLogger(__name__)

# Only import vLLM if available and needed
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.info("vLLM not installed - local model inference not available")


class VLLMLocalClient(BaseLLMClient):
    """
    vLLM client for local model inference
    Use this for self-hosted models like Mistral-7B, Llama, etc.
    
    Requirements:
    - GPU with sufficient VRAM
    - vLLM installed: pip install vllm
    - Model downloaded locally or from HuggingFace
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM requires a GPU and is only needed for local model hosting"
            )
        
        self.model_name = model_name
        self.config = config
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.9),
            max_model_len=config.get('max_model_len', 4096),
            dtype=config.get('dtype', 'auto'),
            download_dir=config.get('download_dir', None)
        )
        
        # Default sampling parameters
        self.default_sampling = SamplingParams(
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.95),
            max_tokens=config.get('max_tokens', 2048)
        )
        
        logger.info(f"Initialized vLLM with model: {model_name}")
    
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using local vLLM"""
        try:
            # Override sampling params if provided
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.default_sampling.temperature),
                top_p=kwargs.get('top_p', self.default_sampling.top_p),
                max_tokens=kwargs.get('max_tokens', self.default_sampling.max_tokens),
                stop=kwargs.get('stop', None)
            )
            
            # Format messages into prompt (model-specific)
            formatted_prompt = self._format_prompt(prompt)
            
            # Run inference
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            # Extract response
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # Calculate token usage
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            
            return {
                "content": generated_text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "model": self.model_name,
                "provider": "vllm_local"
            }
            
        except Exception as e:
            logger.error(f"vLLM generation error: {str(e)}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt for specific model types
        Different models have different prompt formats
        """
        # For Mistral models
        if "mistral" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # For Llama models
        elif "llama" in self.model_name.lower():
            return f"<s>[INST] <<SYS>>\nYou are a helpful research assistant.\n<</SYS>>\n\n{prompt} [/INST]"
        
        # For other models, return as-is
        else:
            return prompt
    
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format messages (not used for vLLM, but required by base class)"""
        return [{"role": "user", "content": prompt}]


class VLLMConfig:
    """Configuration examples for common models"""
    
    MISTRAL_7B = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    LLAMA_7B = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    MIXTRAL_8X7B = {
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpu_memory_utilization": 0.95,  # Requires more VRAM
        "max_model_len": 32768,
        "temperature": 0.7,
        "top_p": 0.95
    }


# Example usage in ai_config.yml:
"""
llm_providers:
  mistral_local:
    enabled: false  # Set to true if you have GPU and want local inference
    type: "vllm"
    model: "mistralai/Mistral-7B-Instruct-v0.2"
    gpu_memory_utilization: 0.9
    max_model_len: 8192
    priority: 5  # Lower priority than API services
    capabilities:
      - local_inference
      - privacy_sensitive
      - high_throughput
"""