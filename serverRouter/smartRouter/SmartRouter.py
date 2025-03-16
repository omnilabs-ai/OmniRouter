"""
SmartRouter: Advanced model selection based on user requirements

This module implements intelligent model routing based on:
1. Task type analysis (coding, math, general knowledge, etc.)
2. Benchmark performance matching
3. Cost-performance-latency balancing
4. User preference weighting
"""

import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import re
from pathlib import Path
import json
import logging
import os
import sys


from serverRouter.core.datamodels import (
    ModelInfo, 
    ChatMessage, 
    SmartRouterRequest,
    ModelProvider,
    BenchmarkScores
)
from serverRouter.core.models import CHAT_MODELS
from serverRouter.smartRouter.embedding_model import OpenAIEmbeddings

TASK_TO_BENCHMARK_WEIGHTS = {
    "coding": {
        "HumanEval": 0.7,
        "BFCL": 0.2,
        "MMLU": 0.1
    },
    "math": {
        "MATH": 0.6,
        "MGSM": 0.3,
        "MMLU": 0.1
    },
    "science": {
        "GPQA": 0.6,
        "MMLU": 0.3,
        "MATH": 0.1
    },
    "reasoning": {
        "BFCL": 0.5,
        "MMLU": 0.3,
        "MGSM": 0.2
    },
    "general_knowledge": {
        "MMLU": 0.8,
        "GPQA": 0.1,
        "BFCL": 0.1
    },
    "creative_writing": {
        "MMLU": 0.5,
        "BFCL": 0.3,
        "HumanEval": 0.2
    }
}

# Keywords for task identification
TASK_KEYWORDS = {
    "coding": [
        "code", "programming", "function", "algorithm", "script", "develop", 
        "javascript", "python", "java", "c++", "ruby", "typescript", "html", "css", 
        "api", "frontend", "backend", "fullstack", "web", "app", "application",
        "debug", "fix", "optimize", "refactor", "implement"
    ],
    "math": [
        "math", "mathematics", "equation", "calculation", "solve", "formula", 
        "algebra", "calculus", "geometry", "statistics", "probability",
        "arithmetic", "numerical", "computation", "mathematical"
    ],
    "science": [
        "physics", "chemistry", "biology", "science", "scientific", "experiment",
        "hypothesis", "theory", "research", "academic", "quantum", "molecular",
        "chemical", "physical", "astronomical", "medical", "technical"
    ],
    "reasoning": [
        "logic", "reasoning", "deduce", "inference", "analyze", "solve", "puzzle", 
        "problem", "critical thinking", "step by step", "explanation", "why",
        "because", "therefore", "conclusion", "premise", "argument", "logical"
    ],
    "general_knowledge": [
        "explain", "what is", "define", "describe", "tell me about", "information",
        "details", "facts", "overview", "summary", "history", "background",
        "context", "knowledge", "general"
    ],
    "creative_writing": [
        "write", "story", "creative", "fiction", "narrative", "character", "plot",
        "novel", "poem", "essay", "article", "content", "blog", "email", "letter",
        "creative", "imaginative", "original", "generate", "create"
    ]
}

class SmartRouter:
    def __init__(self, embeddings_path: str = "serverRouter/smartRouter/benchmark_embeddings.pkl"):
        """Initialize the SmartRouter with benchmark embeddings."""
        self.embeddings_client = OpenAIEmbeddings()
        with open(embeddings_path, 'rb') as f:
            self.benchmark_embeddings = pickle.load(f)
        self.models = CHAT_MODELS
    
    def identify_tasks(self, messages: List[ChatMessage]) -> Dict[str, float]:
        """
        Identify relevant tasks from user messages.
        Returns a dictionary mapping task categories to relevance scores.
        """
        # Extract the latest user message
        user_messages = [msg.content.lower() for msg in messages if msg.role == "user"]
        
        # Use the most recent message with more weight, but consider previous context
        latest_msg = user_messages[-1]
        all_text = " ".join(user_messages)
        
        # Initialize task scores
        task_scores = {task: 0.0 for task in TASK_TO_BENCHMARK_WEIGHTS.keys()}
        
        # Match keywords to identify tasks
        for task, keywords in TASK_KEYWORDS.items():
            # Count keyword matches in latest message (higher weight)
            latest_count = sum(1 for keyword in keywords if keyword.lower() in latest_msg)
            # Count keyword matches in all messages
            total_count = sum(1 for keyword in keywords if keyword.lower() in all_text)
            
            # Calculate weighted score: latest message has 2x the weight
            task_scores[task] = (latest_count * 2 + total_count) / (len(keywords) * 3)
        
        # If no clear task is identified, default to general knowledge
        if all(score < 0.05 for score in task_scores.values()):
            task_scores["general_knowledge"] = 0.5
        
        # Normalize scores to sum to 1.0
        total = sum(task_scores.values())
        if total > 0:
            task_scores = {k: v/total for k, v in task_scores.items()}
        else:
            task_scores = {"general_knowledge": 1.0}
            
        return task_scores

    def compute_benchmark_weights(self, task_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute weights for each benchmark based on identified tasks.
        """
        benchmark_weights = {}
        
        # For each task, add its contribution to benchmark weights
        for task, task_score in task_scores.items():
            if task in TASK_TO_BENCHMARK_WEIGHTS:
                for benchmark, weight in TASK_TO_BENCHMARK_WEIGHTS[task].items():
                    benchmark_weights[benchmark] = benchmark_weights.get(benchmark, 0) + (weight * task_score)
        
        # Normalize benchmark weights
        total = sum(benchmark_weights.values())
        if total > 0:
            benchmark_weights = {k: v/total for k, v in benchmark_weights.items()}
        
        return benchmark_weights

    def score_models(self, 
                     benchmark_weights: Dict[str, float], 
                     rel_cost: float = 0.5, 
                     rel_latency: float = 0.0, 
                     rel_accuracy: float = 0.5,
                     model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Score models based on weighted benchmarks and user preferences.
        
        Args:
            benchmark_weights: Weights for different benchmarks
            rel_cost: Relative importance of cost (0-1)
            rel_latency: Relative importance of latency (0-1)
            rel_accuracy: Relative importance of accuracy (0-1)
            model_names: Optional list of models to consider
            
        Returns:
            Dictionary mapping model names to their scores and performance details
        """
        model_scores = {}
        
        # Normalize preference weights
        pref_total = rel_cost + rel_latency + rel_accuracy
        if pref_total == 0:
            rel_accuracy = 1.0
            pref_total = 1.0
        
        rel_cost = rel_cost / pref_total
        rel_latency = rel_latency / pref_total
        rel_accuracy = rel_accuracy / pref_total
        
        # Filter models if specific names are provided
        available_models = {k: v for k, v in self.models.items() 
                           if model_names is None or k in model_names}
        
        if not available_models:
            return {}
            
        # Find min/max values for normalization
        all_costs = [model.tokenCost for model in available_models.values() if model.tokenCost is not None]
        all_latencies = [model.latency for model in available_models.values() if model.latency is not None]
        
        max_cost = max(all_costs) if all_costs else 1.0
        min_cost = min(all_costs) if all_costs else 0.0
        max_latency = max(all_latencies) if all_latencies else 1.0
        min_latency = min(all_latencies) if all_latencies else 0.0
        
        # Score each model
        for name, model in available_models.items():
            # Skip models without proper metadata
            if not model.benchmarks:
                continue
                
            # Calculate accuracy score based on weighted benchmarks
            accuracy_score = 0.0
            benchmark_coverage = 0.0
            
            for benchmark, weight in benchmark_weights.items():
                if benchmark in model.benchmarks and model.benchmarks[benchmark] is not None:
                    accuracy_score += model.benchmarks[benchmark] * weight
                    benchmark_coverage += weight
            
            # Adjust if some benchmarks are missing
            if benchmark_coverage > 0:
                accuracy_score = accuracy_score / benchmark_coverage
            else:
                accuracy_score = 0.5  # Default if no benchmark data
            
            # Normalize cost (lower is better)
            cost_score = 0.0
            if model.tokenCost is not None:
                if max_cost > min_cost:
                    cost_score = 1.0 - ((model.tokenCost - min_cost) / (max_cost - min_cost))
                else:
                    cost_score = 1.0
            
            # Normalize latency (lower is better)
            latency_score = 0.0
            if model.latency is not None:
                if max_latency > min_latency:
                    latency_score = 1.0 - ((model.latency - min_latency) / (max_latency - min_latency))
                else:
                    latency_score = 1.0
            
            # Combine scores using user preferences
            final_score = (
                accuracy_score * rel_accuracy +
                cost_score * rel_cost +
                latency_score * rel_latency
            )
            
            model_scores[name] = {
                "score": final_score,
                "accuracy": accuracy_score,
                "cost_efficiency": cost_score,
                "speed": latency_score,
                "benchmark_scores": {b: model.benchmarks[b] for b in benchmark_weights if b in model.benchmarks},
                "raw_cost": model.tokenCost,
                "raw_latency": model.latency,
                "provider": model.provider.value
            }
            
        return model_scores

    def select_models(self, request: SmartRouterRequest) -> Dict[str, Any]:
        """
        Select the best models based on user request.
        
        Args:
            request: SmartRouterRequest with messages and preferences
            
        Returns:
            Dictionary with selected models and explanation
        """
        # Identify the type of tasks in the request
        task_scores = self.identify_tasks(request.messages)
        
        # Compute weights for relevant benchmarks
        benchmark_weights = self.compute_benchmark_weights(task_scores)
        
        # Score all models based on weighted benchmarks and user preferences
        model_scores = self.score_models(
            benchmark_weights, 
            rel_cost=request.rel_cost,
            rel_latency=request.rel_latency,
            rel_accuracy=request.rel_accuracy,
            model_names=request.model_names
        )
        
        # Sort models by score
        sorted_models = sorted(
            [(name, data) for name, data in model_scores.items()],
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Get top K models
        top_models = sorted_models[:request.k] if sorted_models else []
        
        # Create explanation for selection
        explanation = self._create_explanation(
            task_scores,
            benchmark_weights,
            top_models,
            request
        )
        
        return {
            "selected_models": [name for name, _ in top_models],
            "model_details": {name: data for name, data in top_models},
            "explanation": explanation if request.verbose else None,
            "identified_tasks": task_scores,
            "benchmark_weights": benchmark_weights
        }
        
    def _create_explanation(self, 
                           task_scores: Dict[str, float],
                           benchmark_weights: Dict[str, float],
                           top_models: List[Tuple[str, Dict[str, Any]]],
                           request: SmartRouterRequest) -> str:
        """Create a detailed explanation of the model selection process."""
        
        # Extract the most relevant tasks
        top_tasks = sorted([(task, score) for task, score in task_scores.items()], 
                          key=lambda x: x[1], reverse=True)[:3]
        
        # Extract the most important benchmarks
        top_benchmarks = sorted([(bench, weight) for bench, weight in benchmark_weights.items()],
                               key=lambda x: x[1], reverse=True)[:3]
        
        explanation = [
            "## Model Selection Explanation",
            "",
            "### Task Analysis",
            f"Based on your request, I identified these key task categories:",
            "".join([f"- {task.replace('_', ' ').title()}: {score:.2f}\n" for task, score in top_tasks]),
            "",
            "### Relevant Benchmarks",
            f"These benchmarks were used to evaluate model performance for your task:",
            "".join([f"- {bench}: {weight:.2f}\n" for bench, weight in top_benchmarks]),
            "",
            "### Selected Models",
            f"Here are the top {len(top_models)} models ranked by overall score:",
            ""
        ]
        
        for i, (name, data) in enumerate(top_models, 1):
            model_info = self.models.get(name)
            if not model_info:
                continue
                
            # Format as percentage
            accuracy = f"{data['accuracy']*100:.1f}%"
            cost_eff = f"{data['cost_efficiency']*100:.1f}%"
            speed = f"{data['speed']*100:.1f}%"
            
            explanation.extend([
                f"#### {i}. {name} ({model_info.provider.value})",
                f"- **Overall Score**: {data['score']:.3f}",
                f"- **Accuracy**: {accuracy}",
                f"- **Cost Efficiency**: {cost_eff}",
                f"- **Speed**: {speed}",
                f"- **Context Length**: {model_info.max_tokens} tokens",
                "",
                f"{model_info.description}",
                ""
            ])
        
        # Add preference weights explanation
        pref_total = request.rel_cost + request.rel_latency + request.rel_accuracy
        norm_cost = request.rel_cost / pref_total if pref_total > 0 else 0
        norm_latency = request.rel_latency / pref_total if pref_total > 0 else 0
        norm_accuracy = request.rel_accuracy / pref_total if pref_total > 0 else 1
        
        explanation.extend([
            "### Preference Weights Used",
            f"- Accuracy Importance: {norm_accuracy:.2f}",
            f"- Cost Importance: {norm_cost:.2f}",
            f"- Speed Importance: {norm_latency:.2f}",
            ""
        ])
        
        return "\n".join(explanation)

    def find_similar_benchmark(self, query_text: str) -> Dict[str, float]:
        """
        Find the benchmarks most similar to the query text.
        Returns a dictionary mapping benchmark names to similarity scores.
        """
        if not self.benchmark_embeddings:
            # Fallback if no embeddings are available
            return {"MMLU": 0.5, "HumanEval": 0.3, "MATH": 0.2}
            
        # Get embedding for query text
        query_embedding = self.embeddings_client.encode(query_text)
        
        # Calculate similarity to each benchmark
        similarities = {}
        for benchmark_id, embedding in self.benchmark_embeddings.items():
            similarity = self.embeddings_client.similarity(query_embedding, embedding)
            similarities[benchmark_id] = similarity
            
        # Normalize similarities
        total = sum(similarities.values())
        if total > 0:
            similarities = {k: v/total for k, v in similarities.items()}
            
        return similarities
    
    def get_top_user_models(self, query: str, k: int = 5, model_names=None, 
                       rel_cost: float = 0.5, rel_latency: float = 0.0, 
                       rel_accuracy: float = 0.5) -> dict:
        """
        Get the top models for a user query based on the query content and user preferences.
        
        Args:
            query: The user query text
            k: Number of top models to return (default 5)
            model_names: Optional list of model names to consider, if None uses all models
            rel_cost: Relative importance of cost (0-1)
            rel_latency: Relative importance of latency (0-1)
            rel_accuracy: Relative importance of accuracy (0-1)
            
        Returns:
            Dictionary with the selected model and details
        """
        # Create a dummy message for the task identification
        messages = [ChatMessage(role="user", content=query)]


        # Identify the type of tasks in the query
        task_scores = self.identify_tasks(messages)
        
        # Compute weights for relevant benchmarks
        benchmark_weights = self.compute_benchmark_weights(task_scores)
        
        # Score all models based on weighted benchmarks and user preferences
        model_scores = self.score_models(
            benchmark_weights, 
            rel_cost=rel_cost,
            rel_latency=rel_latency,
            rel_accuracy=rel_accuracy,
            model_names=model_names
        )
        
        # Sort models by score
        sorted_models = sorted(
            [(name, data) for name, data in model_scores.items()],
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Get top K models
        top_models = sorted_models[:k] if sorted_models else []
        
        # Return the top model with some details (can be expanded as needed)
        if top_models:
            top_model, top_model_data = top_models[0]
            return {
                "model": top_model,
                "model_details": top_model_data,
                "identified_tasks": task_scores,
                "benchmark_weights": benchmark_weights
            }
        else:
            # Fallback to a default model if no models found
            return {
                "model": "gpt-3.5-turbo",  # Default model as fallback
                "model_details": {},
                "identified_tasks": task_scores,
                "benchmark_weights": benchmark_weights
            }