"""
Enhanced embedding model for vector similarity calculations.

This module provides functionality to:
1. Generate embeddings using OpenAI's API
2. Calculate cosine similarity between embeddings
3. Cache embeddings for better performance
"""

from openai import OpenAI
import numpy as np
from typing import Dict, Optional, List, Union
import os
from pathlib import Path
import json
import time

class OpenAIEmbeddings:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "text-embedding-3-small",
                 cache_dir: Optional[str] = None):
        """
        Initialize OpenAI embeddings client with optional caching.
        
        Args:
            api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY env variable
            model: OpenAI embedding model to use
            cache_dir: Directory to store cached embeddings. If None, caching is disabled
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = {}
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self.api_calls = 0
        
        # Load cache if directory is provided
        if cache_dir:
            self._load_cache(cache_dir)
    
    def _load_cache(self, cache_dir: str) -> None:
        """Load cached embeddings from disk."""
        cache_path = Path(cache_dir) / f"embeddings_cache_{self.model.replace('-', '_')}.json"
        
        if not cache_path.exists():
            return
            
        try:
            with open(cache_path, 'r') as f:
                # The cache stores text -> embedding as a dictionary
                # Convert lists back to numpy arrays
                cached_data = json.load(f)
                self.cache = {k: np.array(v) for k, v in cached_data.items()}
        except Exception as e:
            return
    def _save_cache(self) -> None:
        """Save cached embeddings to disk."""
        if not self.cache_dir:
            return
            
        cache_path = Path(self.cache_dir) / f"embeddings_cache_{self.model.replace('-', '_')}.json"
        
        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {k: v.tolist() for k, v in self.cache.items()}
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            return
        except Exception as e:
            return
    
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text into vector embeddings.
        
        Args:
            text: Text string or list of strings to encode
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        if isinstance(text, list):
            # Handle batch encoding
            return self.batch_encode(text)
        
        # Check if text is in cache
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        # Rate limiting: simple exponential backoff
        max_retries = 5
        retry = 0
        
        while retry < max_retries:
            try:
                self.api_calls += 1
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                embedding = np.array(response.data[0].embedding)
                
                # Cache the result
                self.cache[text] = embedding
                
                # Save cache periodically (every 10 new entries)
                if self.cache_dir and len(self.cache) % 10 == 0:
                    self._save_cache()
                    
                return embedding
                
            except Exception as e:
                retry += 1
                wait_time = 2 ** retry  # Exponential backoff
                
                if retry < max_retries:
                    time.sleep(wait_time)
                else:
                    # Return a zero vector as fallback
                    dims = 1536 if "ada" in self.model else 3072 if "3" in self.model else 1536
                    return np.zeros(dims)
    
    def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts in an efficient batch.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors
        """
        # Check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self.cache:
                self.cache_hits += 1
                results[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all texts were cached, return results
        if not uncached_texts:
            return results
        
        # Otherwise, get embeddings for uncached texts
        try:
            self.api_calls += 1
            response = self.client.embeddings.create(
                model=self.model,
                input=uncached_texts,
                encoding_format="float"
            )
            
            # Process and cache responses
            for i, embedding_data in enumerate(response.data):
                text_idx = uncached_indices[i]
                embedding = np.array(embedding_data.embedding)
                
                # Cache the result
                self.cache[texts[text_idx]] = embedding
                results[text_idx] = embedding
            
            # Save cache periodically
            if self.cache_dir and len(uncached_texts) > 0:
                self._save_cache()
                
            return results
            
        except Exception as e:
            # Return zero vectors as fallback
            dims = 1536 if "ada" in self.model else 3072 if "3" in self.model else 1536
            zero_vector = np.zeros(dims)
            
            for idx in uncached_indices:
                results[idx] = zero_vector
                
            return results
            
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Check for zero vectors to avoid division by zero
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Compute normalized dot product
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        return float(np.dot(embedding1_normalized, embedding2_normalized))
    
    def similarity_matrix(self, embeddings1: List[np.ndarray], embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First list of embedding vectors
            embeddings2: Second list of embedding vectors
            
        Returns:
            Matrix of similarity scores where matrix[i][j] is the 
            similarity between embeddings1[i] and embeddings2[j]
        """
        # Normalize all embeddings
        normalized1 = np.array([
            emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else np.zeros_like(emb)
            for emb in embeddings1
        ])
        
        normalized2 = np.array([
            emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else np.zeros_like(emb)
            for emb in embeddings2
        ])
        
        # Compute similarity matrix using dot product
        return np.dot(normalized1, normalized2.T)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage and API calls."""
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls
        }