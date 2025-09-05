#!/usr/bin/env python3
"""
Model Clients for Phase Evaluation Engine
Handles API calls to different LLM providers
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
import google.generativeai as genai
from groq import Groq
import anthropic
import requests
from config.model_config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, 
    ANTHROPIC_API_KEY, XAI_API_KEY, MODEL_LIMITS
)

class ModelClient:
    """Base class for model clients"""
    
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = MODEL_LIMITS.get(model_name, {}).get("max_tokens", 4096)
        
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call the model with a prompt"""
        raise NotImplementedError

class OpenAIClient(ModelClient):
    """OpenAI API client"""
    
    def __init__(self, model_name: str):
        super().__init__("openai", model_name)
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except TypeError as e:
            # Handle older OpenAI client versions
            if "proxies" in str(e):
                import httpx
                # Create a custom httpx client without proxies
                http_client = httpx.Client()
                self.client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            else:
                raise e
    
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Handle GPT-5 models which use responses.create() instead of chat.completions.create()
            if self.model_name.startswith("gpt-5"):
                # GPT-5 models use the new responses API with minimal parameters
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    reasoning={"effort": "medium"},  # GPT-5 specific: minimal, low, medium, high
                    text={"verbosity": "medium"}     # GPT-5 specific: low, medium, high
                )
            else:
                # Older models use max_tokens
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", 0.1),
                    timeout=kwargs.get("timeout", 120)
                )
            
            end_time = time.time()
            
            # Handle different response structures for GPT-5 vs other models
            if self.model_name.startswith("gpt-5"):
                # GPT-5 uses responses API with different structure
                return {
                    "success": True,
                    "response": response.output_text,
                    "duration": end_time - start_time,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                        "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
                    }
                }
            else:
                # Standard chat completions API
                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "duration": end_time - start_time,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

class GoogleClient(ModelClient):
    """Google Gemini API client"""
    
    def __init__(self, model_name: str):
        super().__init__("google", model_name)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)
    
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", 0.1)
                )
            )
            
            end_time = time.time()
            
            return {
                "success": True,
                "response": response.text,
                "duration": end_time - start_time,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

class GroqClient(ModelClient):
    """Groq API client"""
    
    def __init__(self, model_name: str):
        super().__init__("groq", model_name)
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", 0.1)
            )
            
            end_time = time.time()
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "duration": end_time - start_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

class AnthropicClient(ModelClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model_name: str):
        super().__init__("anthropic", model_name)
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", 0.1),
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = time.time()
            
            return {
                "success": True,
                "response": response.content[0].text,
                "duration": end_time - start_time,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

class XAIClient(ModelClient):
    """x.ai Grok API client"""
    
    def __init__(self, model_name: str):
        super().__init__("xai", model_name)
        self.api_key = XAI_API_KEY
        self.base_url = "https://api.x.ai/v1"
    
    def call_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", 0.1)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 300)
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "duration": end_time - start_time,
                    "usage": {
                        "prompt_tokens": result["usage"]["prompt_tokens"],
                        "completion_tokens": result["usage"]["completion_tokens"],
                        "total_tokens": result["usage"]["total_tokens"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code} - {response.text}",
                    "duration": end_time - start_time,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

def get_model_client(provider: str, model_name: str) -> Optional[ModelClient]:
    """Get the appropriate model client for a provider and model"""
    if provider == "openai" and OPENAI_API_KEY:
        return OpenAIClient(model_name)
    elif provider == "google" and GOOGLE_API_KEY:
        return GoogleClient(model_name)
    elif provider == "groq" and GROQ_API_KEY:
        return GroqClient(model_name)
    elif provider == "anthropic" and ANTHROPIC_API_KEY:
        return AnthropicClient(model_name)
    elif provider == "xai" and XAI_API_KEY:
        return XAIClient(model_name)
    else:
        return None

def call_model_with_retry(provider: str, model_name: str, prompt: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
    """Call a model with retry logic"""
    client = get_model_client(provider, model_name)
    if not client:
        return {
            "success": False,
            "error": f"No client available for {provider}/{model_name}",
            "duration": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    for attempt in range(max_retries):
        result = client.call_model(prompt, **kwargs)
        if result["success"]:
            return result
        elif attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return result
