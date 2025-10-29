

import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Note: .env has GROK_API_KEY, not GROQ_API_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")  # For x.ai models

# Model context limits and token management
MODEL_LIMITS = {
    # OpenAI models
    "gpt-4": {"max_tokens": 8192, "context_length": 8192},
    "gpt-3.5-turbo": {"max_tokens": 4096, "context_length": 16385},
    "gpt-4o": {"max_tokens": 4096, "context_length": 128000},
    "gpt-5": {"max_tokens": 4000, "context_length": 128000},  # GPT-5 has advanced long-context capabilities
    
    # Google models
    "gemini-1.5-flash": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-pro": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-flash-latest": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-pro-latest": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-flash-002": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-pro-002": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-1.5-flash-8b": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.0-flash": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.0-flash-001": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.0-flash-lite": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.0-flash-lite-001": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.5-flash": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.5-pro": {"max_tokens": 8192, "context_length": 1000000},
    "gemini-2.5-flash-lite": {"max_tokens": 8192, "context_length": 1000000},
    
    # Gemma models
    "gemma-3-1b-it": {"max_tokens": 8192, "context_length": 8192},
    "gemma-3-4b-it": {"max_tokens": 8192, "context_length": 8192},
    "gemma-3-12b-it": {"max_tokens": 8192, "context_length": 8192},
    "gemma-3-27b-it": {"max_tokens": 8192, "context_length": 8192},
    "gemma-3n-e4b-it": {"max_tokens": 8192, "context_length": 8192},
    "gemma-3n-e2b-it": {"max_tokens": 8192, "context_length": 8192},
    
    # Groq models
    "llama3-8b-8192": {"max_tokens": 4096, "context_length": 8192},
    "llama3-70b-8192": {"max_tokens": 4096, "context_length": 8192},
    "mixtral-8x7b-32768": {"max_tokens": 4096, "context_length": 32768},
    
    # Anthropic models
    "claude-3-opus-20240229": {"max_tokens": 4096, "context_length": 200000},
    "claude-3-sonnet-20240229": {"max_tokens": 4096, "context_length": 200000},
    "claude-3-haiku-20240307": {"max_tokens": 4096, "context_length": 200000},
    
    # x.ai models
    "grok-2-1212": {"max_tokens": 16384, "context_length": 32768},
    "grok-2-vision-1212": {"max_tokens": 16384, "context_length": 32768},
    "grok-3": {"max_tokens": 16384, "context_length": 32768}
}

# Cost per 1K tokens (approximate)
COST_PER_1K_TOKENS = {
    # OpenAI models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-5": {"input": 0.0015625, "output": 0.0125},  # Official GPT-5 pricing: $1.5625 per 1M input, $12.50 per 1M output (converted to per 1K)


    # Google models
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00375, "output": 0.015},
    "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.5-pro": {"input": 0.00375, "output": 0.015},
    
    # Groq models
    "llama3-8b-8192": {"input": 0.00005, "output": 0.0001},
    "llama3-70b-8192": {"input": 0.00059, "output": 0.0008},
    "mixtral-8x7b-32768": {"input": 0.00014, "output": 0.00024},
    
    # Anthropic models
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    
    # x.ai models (approximate)
    "grok-2-1212": {"input": 0.001, "output": 0.002},
    "grok-2-vision-1212": {"input": 0.001, "output": 0.002},
    "grok-3": {"input": 0.002, "output": 0.004}
}

# Comprehensive model configurations
MODELS = {
    "openai": {
        "gpt-4": {"provider": "openai", "model": "gpt-4"},
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
        "gpt-5": {"provider": "openai", "model": "gpt-5"}
    },
    "google": {
        # Gemini 1.5 models
        "gemini-1.5-flash": {"provider": "google", "model": "gemini-1.5-flash"},
        "gemini-1.5-pro": {"provider": "google", "model": "gemini-1.5-pro"},
        "gemini-1.5-flash-latest": {"provider": "google", "model": "gemini-1.5-flash-latest"},
        "gemini-1.5-pro-latest": {"provider": "google", "model": "gemini-1.5-pro-latest"},
        "gemini-1.5-flash-002": {"provider": "google", "model": "gemini-1.5-flash-002"},
        "gemini-1.5-pro-002": {"provider": "google", "model": "gemini-1.5-pro-002"},
        "gemini-1.5-flash-8b": {"provider": "google", "model": "gemini-1.5-flash-8b"},
        
        # Gemini 2.0 models
        "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash"},
        "gemini-2.0-flash-001": {"provider": "google", "model": "gemini-2.0-flash-001"},
        "gemini-2.0-flash-lite": {"provider": "google", "model": "gemini-2.0-flash-lite"},
        "gemini-2.0-flash-lite-001": {"provider": "google", "model": "gemini-2.0-flash-lite-001"},
        
        # Gemini 2.5 models
        "gemini-2.5-flash": {"provider": "google", "model": "gemini-2.5-flash"},
        "gemini-2.5-pro": {"provider": "google", "model": "gemini-2.5-pro"},
        "gemini-2.5-flash-lite": {"provider": "google", "model": "gemini-2.5-flash-lite"},
        
        # Gemma models
        "gemma-3-1b-it": {"provider": "google", "model": "gemma-3-1b-it"},
        "gemma-3-4b-it": {"provider": "google", "model": "gemma-3-4b-it"},
        "gemma-3-12b-it": {"provider": "google", "model": "gemma-3-12b-it"},
        "gemma-3-27b-it": {"provider": "google", "model": "gemma-3-27b-it"},
        "gemma-3n-e4b-it": {"provider": "google", "model": "gemma-3n-e4b-it"},
        "gemma-3n-e2b-it": {"provider": "google", "model": "gemma-3n-e2b-it"}
    },
    "groq": {
        "llama3-8b-8192": {"provider": "groq", "model": "llama3-8b-8192"},
        "llama3-70b-8192": {"provider": "groq", "model": "llama3-70b-8192"},
        "mixtral-8x7b-32768": {"provider": "groq", "model": "mixtral-8x7b-32768"}
    },
    "anthropic": {
        "claude-3-opus-20240229": {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        "claude-3-sonnet-20240229": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
        "claude-3-haiku-20240307": {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
    },
    "xai": {
        "grok-2-1212": {"provider": "xai", "model": "grok-2-1212"},
        "grok-2-vision-1212": {"provider": "xai", "model": "grok-2-vision-1212"},
        "grok-3": {"provider": "xai", "model": "grok-3"}
    }
}

def get_available_models() -> Dict[str, List[str]]:
    """Get list of models available based on API keys"""
    available = {}
    
    if OPENAI_API_KEY:
        available["openai"] = list(MODELS["openai"].keys())
    
    if GOOGLE_API_KEY:
        available["google"] = list(MODELS["google"].keys())
    
    # Note: .env file has GROK_API_KEY instead of GROQ_API_KEY
    # For now, we'll skip Groq models since the key is missing
    if GROQ_API_KEY:
        available["groq"] = list(MODELS["groq"].keys())
    
    if ANTHROPIC_API_KEY:
        available["anthropic"] = list(MODELS["anthropic"].keys())
    
    if XAI_API_KEY or GROK_API_KEY:
        available["xai"] = list(MODELS["xai"].keys())
    
    return available

def get_model_config(provider: str, model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if provider in MODELS and model_name in MODELS[provider]:
        config = MODELS[provider][model_name].copy()
        config.update(MODEL_LIMITS.get(model_name, {}))
        return config
    return {}

def get_model_cost(provider: str, model_name: str) -> Dict[str, float]:
    """Get cost configuration for a specific model"""
    return COST_PER_1K_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
