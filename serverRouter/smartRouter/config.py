"""
Configuration for the SmartRouter system.

This module contains configuration settings for the SmartRouter, including:
- File paths
- Thresholds and weights
- Provider adjustments
- Task-specific settings
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import os

class SmartRouterConfig:
    """Configuration for SmartRouter components and behavior."""
    
    def __init__(self):
        """Initialize default configuration values."""
        # File paths
        self.embeddings_path = "serverRouter/smartRouter/benchmark_embeddings.pkl"
        self.task_db_path = "serverRouter/smartRouter/task_examples_db.pkl"
        
        # Task identification settings
        self.task_identification_method = "hybrid"  # Options: "keywords", "vector_db", "hybrid"
        self.vector_similarity_threshold = 0.6  # Threshold for vector similarity in task classification
        self.vector_weight = 0.7  # Weight given to vector-based classification in hybrid approach
        self.keyword_weight = 0.3  # Weight given to keyword-based classification in hybrid approach
        
        # Provider adjustment factors for diversity
        self.provider_adjustments = {
            "gemini": -0.1,     # Slight penalty for Gemini models to counter their dominance
            "google": -0.1,     # Slight penalty for Google models (same as gemini)
            "anthropic": 0.1,   # Boost for Claude models
            "openai": 0.1,      # Boost for OpenAI models
            "deepseek": 0.1,    # Boost for DeepSeek models
            "together": 0.1     # Boost for Together models
        }
        
        # Task-specific model adjustments
        self.creative_writing_adjustments = {
            "penalty": {
                "deepseek": 0.25  # Penalty for DeepSeek on creative writing
            },
            "boost": {
                "anthropic": 0.15,  # Boost for Anthropic on creative writing
                "openai": 0.15      # Boost for OpenAI on creative writing
            },
            "excluded_providers": ["deepseek"]  # Providers to exclude for creative writing
        }
        
        # Cache settings
        self.enable_caching = True
        self.cache_ttl = 300  # 5 minutes cache time-to-live
        
    def get_file_path(self, path_name: str) -> str:
        """Get a file path from the configuration with environment variable support."""
        if path_name == "embeddings":
            path = self.embeddings_path
        elif path_name == "task_db":
            path = self.task_db_path
        else:
            raise ValueError(f"Unknown path name: {path_name}")
            
        # Check for environment variable override
        env_var = f"SMART_ROUTER_{path_name.upper()}_PATH"
        if env_var in os.environ:
            return os.environ[env_var]
        
        return path
        
    def get_provider_adjustment(self, provider: str) -> float:
        """Get the adjustment factor for a provider."""
        return self.provider_adjustments.get(provider, 0.0)
        
    def get_task_provider_adjustment(self, task: str, provider: str) -> float:
        """Get task-specific provider adjustment."""
        if task == "creative_writing":
            # Apply penalty
            if provider in self.creative_writing_adjustments["penalty"]:
                return -self.creative_writing_adjustments["penalty"][provider]
            
            # Apply boost
            if provider in self.creative_writing_adjustments["boost"]:
                return self.creative_writing_adjustments["boost"][provider]
        
        return 0.0
        
    def should_exclude_provider(self, task: str, provider: str) -> bool:
        """Check if a provider should be excluded for a specific task."""
        if task == "creative_writing":
            return provider in self.creative_writing_adjustments["excluded_providers"]
        
        return False

# Default configuration instance
CONFIG = SmartRouterConfig()