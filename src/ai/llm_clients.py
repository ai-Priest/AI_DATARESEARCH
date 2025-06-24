"""
LLM Client implementations for MiniMax, Mistral, Claude, and OpenAI
Handles API interactions with fallback logic and error handling
"""
import os
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Import official SDKs
import anthropic
import openai
from mistralai import Mistral, SystemMessage, UserMessage, AssistantMessage
from mistralai import ToolMessage, ChatCompletionRequest, ChatCompletionResponse
import aiohttp  # Still needed for MiniMax as they don't have official Python SDK

# Configure logging
logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for all LLM clients"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        self.model = config.get('model')
        self.api_base = config.get('api_base')
        self.max_tokens = config.get('max_tokens', 2048)
        self.temperature = config.get('temperature', 0.7)
        self.timeout = config.get('timeout', 30)
        
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion for given prompt"""
        pass
    
    @abstractmethod
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt into messages for the API"""
        pass


class MiniMaxClient(BaseLLMClient):
    """MiniMax API client implementation"""
    
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using MiniMax API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": self.format_messages(prompt),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle MiniMax's different response format
                        if 'base_resp' in result:
                            # Check for errors in base_resp
                            base_resp = result['base_resp']
                            if base_resp.get('status_code') != 0:
                                raise Exception(f"MiniMax API error: {base_resp.get('status_msg', 'Unknown error')}")
                        
                        # Try standard OpenAI format first
                        if 'choices' in result and result['choices']:
                            content = result['choices'][0]['message']['content']
                        # Try MiniMax specific format
                        elif 'reply' in result:
                            content = result['reply']
                        # Try alternative format
                        elif 'output' in result:
                            content = result['output']
                        else:
                            logger.warning(f"Unexpected MiniMax response format: {result}")
                            raise Exception("Unable to extract content from MiniMax response")
                            
                        return {
                            "content": content,
                            "usage": result.get('usage', {}),
                            "model": self.model,
                            "provider": "minimax"
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"MiniMax API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"MiniMax completion error: {str(e)}")
            raise
    
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt for MiniMax API"""
        return [
            {"role": "system", "content": "You are an expert research assistant specializing in dataset discovery and academic methodology."},
            {"role": "user", "content": prompt}
        ]


class MistralClient(BaseLLMClient):
    """Mistral API client implementation using official SDK"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        super().__init__(api_key, config)
        self.client = Mistral(api_key=api_key)
    
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using Mistral API"""
        try:
            # Format messages
            messages = [
                SystemMessage(content="You are a technical expert in data analysis and research methodology."),
                UserMessage(content=prompt)
            ]
            
            # Make API call using official SDK
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": self.model,
                "provider": "mistral"
            }
                        
        except Exception as e:
            logger.error(f"Mistral completion error: {str(e)}")
            raise
    
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt for Mistral API"""
        return [
            {"role": "system", "content": "You are a technical expert in data analysis and research methodology."},
            {"role": "user", "content": prompt}
        ]


class ClaudeClient(BaseLLMClient):
    """Claude API client implementation using official SDK"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        super().__init__(api_key, config)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using Claude API"""
        try:
            # Use async client for better performance
            async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await async_client.messages.create(
                model=self.model,
                messages=self.format_messages(prompt),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return {
                "content": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "model": self.model,
                "provider": "claude"
            }
                        
        except Exception as e:
            logger.error(f"Claude completion error: {str(e)}")
            raise
    
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt for Claude API"""
        return [{"role": "user", "content": prompt}]


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation using official SDK"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        super().__init__(api_key, config)
        # Initialize OpenAI client
        openai.api_key = api_key
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
    
    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using OpenAI API"""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=self.format_messages(prompt),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": self.model,
                "provider": "openai"
            }
                        
        except Exception as e:
            logger.error(f"OpenAI completion error: {str(e)}")
            raise
    
    def format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt for OpenAI API"""
        return [
            {"role": "system", "content": "You are a helpful research assistant specializing in dataset discovery."},
            {"role": "user", "content": prompt}
        ]


class LLMManager:
    """Manages multiple LLM clients with fallback logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients: Dict[str, BaseLLMClient] = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize all enabled LLM clients"""
        providers = self.config.get('llm_providers', {})
        
        # Initialize MiniMax
        if providers.get('minimax', {}).get('enabled'):
            api_key = os.getenv('MINIMAX_API_KEY')
            if api_key:
                self.clients['minimax'] = MiniMaxClient(api_key, providers['minimax'])
                
        # Initialize Mistral
        if providers.get('mistral', {}).get('enabled'):
            api_key = os.getenv('MISTRAL_API_KEY')
            if api_key:
                self.clients['mistral'] = MistralClient(api_key, providers['mistral'])
                
        # Initialize Claude
        if providers.get('claude', {}).get('enabled'):
            api_key = os.getenv('CLAUDE_API_KEY')
            if api_key:
                self.clients['claude'] = ClaudeClient(api_key, providers['claude'])
                
        # Initialize OpenAI
        if providers.get('openai', {}).get('enabled'):
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.clients['openai'] = OpenAIClient(api_key, providers['openai'])
                
        logger.info(f"Initialized LLM clients: {list(self.clients.keys())}")
    
    async def complete_with_fallback(
        self, 
        prompt: str, 
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic fallback to other providers
        
        Args:
            prompt: The prompt to complete
            preferred_provider: Preferred provider to try first
            **kwargs: Additional arguments for the completion
            
        Returns:
            Completion result with content and metadata
        """
        # Determine provider order
        providers = self._get_provider_order(preferred_provider)
        
        last_error = None
        for provider in providers:
            if provider not in self.clients:
                continue
                
            try:
                logger.info(f"Attempting completion with {provider}")
                start_time = time.time()
                
                result = await self.clients[provider].complete(prompt, **kwargs)
                
                # Add timing information
                result['response_time'] = time.time() - start_time
                result['provider_used'] = provider
                
                logger.info(f"Successfully completed with {provider} in {result['response_time']:.2f}s")
                return result
                
            except Exception as e:
                logger.warning(f"Failed with {provider}: {str(e)}")
                last_error = e
                continue
        
        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {str(last_error)}")
    
    def _get_provider_order(self, preferred_provider: Optional[str] = None) -> List[str]:
        """Get provider order based on priority and preference"""
        providers = self.config.get('llm_providers', {})
        
        # Sort by priority
        sorted_providers = sorted(
            [(name, cfg) for name, cfg in providers.items() if cfg.get('enabled')],
            key=lambda x: x[1].get('priority', 999)
        )
        
        provider_order = [name for name, _ in sorted_providers]
        
        # Move preferred provider to front if specified
        if preferred_provider and preferred_provider in provider_order:
            provider_order.remove(preferred_provider)
            provider_order.insert(0, preferred_provider)
            
        return provider_order
    
    async def complete_for_capability(
        self,
        prompt: str,
        capability: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete using a provider that has the specified capability
        
        Args:
            prompt: The prompt to complete
            capability: Required capability (e.g., 'research_methodology')
            **kwargs: Additional arguments
            
        Returns:
            Completion result
        """
        # Find providers with the capability
        capable_providers = []
        providers = self.config.get('llm_providers', {})
        
        for name, cfg in providers.items():
            if capability in cfg.get('capabilities', []) and name in self.clients:
                capable_providers.append((name, cfg.get('priority', 999)))
        
        # Sort by priority
        capable_providers.sort(key=lambda x: x[1])
        provider_order = [name for name, _ in capable_providers]
        
        if not provider_order:
            # Fallback to any available provider
            return await self.complete_with_fallback(prompt, **kwargs)
        
        # Try capable providers first
        last_error = None
        for provider in provider_order:
            try:
                return await self.clients[provider].complete(prompt, **kwargs)
            except Exception as e:
                last_error = e
                continue
        
        # Fallback to general providers if all capable ones fail
        return await self.complete_with_fallback(prompt, **kwargs)