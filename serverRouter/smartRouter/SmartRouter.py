from sentence_transformers import SentenceTransformer
from serverRouter.core.models import CHAT_MODELS
import pickle
import os
from typing import Dict, Union, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class RouterDecision:
    """Class to store and format the router's decision-making process."""
    benchmark_similarities: Dict[str, float]
    model_accuracies: Dict[str, Dict[str, float]]
    normalized_metrics: Dict[str, Dict[str, float]]
    final_scores: Dict[str, float]
    chosen_model: str

    def format_verbose_output(self) -> str:
        """Format the decision process into a detailed string."""
        output = []
        
        # Benchmark similarities
        output.append("=== Benchmark Similarities ===")
        sorted_benchmarks = sorted(self.benchmark_similarities.items(), key=lambda x: x[1], reverse=True)
        for benchmark, score in sorted_benchmarks:
            output.append(f"{benchmark}: {score:.4f}")
            
        # Model metrics
        output.append("\n=== Model Metrics ===")
        for model in self.model_accuracies:
            output.append(f"\n{model}:")
            output.append(f"  Raw Accuracy: {self.model_accuracies[model]['accuracy']:.4f}")
            output.append(f"  Raw Cost: {self.model_accuracies[model]['cost']:.4f}")
            output.append(f"  Raw Latency: {self.model_accuracies[model]['latency']:.4f}")
            
        # Normalized metrics
        output.append("\n=== Normalized Metrics ===")
        for metric, scores in self.normalized_metrics.items():
            output.append(f"\n{metric.title()}:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for model, score in sorted_scores:
                output.append(f"  {model}: {score:.4f}")
                
        # Final decision
        output.append("\n=== Final Weighted Scores ===")
        sorted_final = sorted(self.final_scores.items(), key=lambda x: x[1], reverse=True)
        for model, score in sorted_final:
            output.append(f"{model}: {score:.4f}")
            
        output.append(f"\nChosen Model: {self.chosen_model}")
        
        return "\n".join(output)

class SmartRouter:
    def __init__(self, embeddings_file: str = "serverRouter/smartRouter/benchmark_embeddings.pkl", verbose: bool = False) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.benchmark_embeddings: Dict[str, np.ndarray] = {}
        self.verbose = verbose
        
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                self.benchmark_embeddings = pickle.load(f)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a vector embedding."""
        return self.model.encode(text)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return self.model.similarity(embedding1, embedding2)

    def get_benchmark_similarities(self, query: str) -> Dict[str, float]:
        """Get similarity scores for all benchmarks compared to the query text.
        
        Args:
            query (str): The query text to compare against benchmarks
            
        Returns:
            Dict[str, float]: Dictionary mapping benchmark names to their normalized similarity scores (sum = 1)
        """
        query_embedding = self.encode(query)
        similarities = {}
        
        # Calculate raw similarities
        for benchmark_id, benchmark_embedding in self.benchmark_embeddings.items():
            similarity = self.compute_similarity(query_embedding, benchmark_embedding)
            similarities[benchmark_id] = max(0, float(similarity))
        
        # Normalize scores to sum to 1
        total_similarity = sum(similarities.values())
        if total_similarity > 0:  # Avoid division by zero
            for benchmark_id in similarities:
                similarities[benchmark_id] = similarities[benchmark_id] / total_similarity
        
        return similarities

    def get_top_accuracy_models(self, benchmark_weights, k: int = 3, model_names: list[str] | None = None) -> list[tuple[str, float]]:
        """Get the top k best models for the query based on benchmark similarities.
        
        Args:
            query (str): The query text to compare against benchmarks
            k (int): Number of top models to return
            model_names (list[str] | None): Optional list of model names to select from. If None, all models are considered.
            
        Returns:
            list[tuple[str, float]]: List of (model_name, score) tuples, sorted by score in descending order
        """
        # Calculate weighted scores for each model
        model_scores = {}
        for model_name, model_info in CHAT_MODELS.items():
            # Skip if model is not in the provided list of models (if specified)
            if model_names is not None and model_name not in model_names:
                continue
                
            if not hasattr(model_info, 'benchmarks') or not model_info.benchmarks:
                continue
            
            weighted_score = 0.0
            for benchmark, weight in benchmark_weights.items():
                if benchmark in model_info.benchmarks:
                    weighted_score += model_info.benchmarks[benchmark] * weight
            
            model_scores[model_name] = {
                "accuracy": weighted_score,
                "cost": model_info.tokenCost,
                "latency": model_info.latency,
            }
        
        # Sort models by score and get top k
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        return sorted_models[:k]

    def _normalize_dict_values(self, d: Dict[str, float]) -> Dict[str, float]:
        """Normalize dictionary values to sum to 1."""
        total = sum(d.values())
        if total <= 0:
            return {k: 0.0 for k in d.keys()}
        return {k: v/total for k, v in d.items()}

    def _invert_and_normalize(self, d: Dict[str, float]) -> Dict[str, float]:
        """Invert values (1/x) and normalize to sum to 1."""
        inverted = {k: 1/v for k, v in d.items()}
        return self._normalize_dict_values(inverted)

    def _calculate_model_metrics(self, models: list[tuple[str, dict]], metric_weights: dict) -> Union[str, Tuple[str, RouterDecision]]:
        """Calculate final scores for models based on weighted metrics."""
        # Extract and normalize individual metrics
        metrics = {
            'accuracy': {m[0]: m[1]['accuracy'] for m in models},
            'cost': {m[0]: m[1]['cost'] for m in models},
            'latency': {m[0]: m[1]['latency'] for m in models}
        }
        
        normalized_metrics = {
            'accuracy': self._normalize_dict_values(metrics['accuracy']),
            'cost': self._invert_and_normalize(metrics['cost']),
            'latency': self._invert_and_normalize(metrics['latency'])
        }
        
        # Calculate weighted scores
        scores = {}
        for model_name in normalized_metrics['accuracy'].keys():
            scores[model_name] = sum(
                normalized_metrics[metric][model_name] * weight
                for metric, weight in metric_weights.items()
            )
            
        # Normalize final scores
        normalized_scores = self._normalize_dict_values(scores)
        chosen_model = max(normalized_scores.items(), key=lambda x: x[1])[0]
        
        # Create decision object for verbose output
        model_metrics = {m[0]: m[1] for m in models}
        decision = RouterDecision(
            benchmark_similarities=self.last_benchmark_weights,
            model_accuracies=model_metrics,
            normalized_metrics=normalized_metrics,
            final_scores=normalized_scores,
            chosen_model=chosen_model
        )
        
        if self.verbose:
            return chosen_model, decision
        return chosen_model

    def get_top_user_models(
        self, 
        query: str, 
        k: int = 5, 
        model_names: list[str] | None = None, 
        rel_cost: float = 0.5, 
        rel_latency: float = 0, 
        rel_accuracy: float = 0.5
    ) -> Union[str, dict]:
        """Get the best model based on accuracy, cost, and latency preferences.
        
        Returns:
            If verbose=False: str - The name of the chosen model
            If verbose=True: dict - A dictionary containing:
                - 'model': The name of the chosen model
                - 'explanation': A detailed explanation of the decision process
        """
        benchmark_weights = self.get_benchmark_similarities(query)
        self.last_benchmark_weights = benchmark_weights  # Store for verbose output
        best_models = self.get_top_accuracy_models(benchmark_weights, k, model_names)
        
        metric_weights = {
            'accuracy': rel_accuracy,
            'cost': rel_cost,
            'latency': rel_latency
        }
        
        result = self._calculate_model_metrics(best_models, metric_weights)
        
        if self.verbose:
            chosen_model, decision = result
            return {
                'model': chosen_model,
                'explanation': decision.format_verbose_output()
            }
        return result