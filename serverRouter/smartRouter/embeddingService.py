from abc import ABC, abstractmethod
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingServiceInterface(ABC):
    """
    Interface for embedding services to ensure consistent API across different providers.
    """
    
    @abstractmethod
    def generate_embedding(self, text):
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts):
        """Generate embeddings for multiple texts"""
        pass
    
    def calculate_similarities(self, query_embedding, embedding_dict):
        """
        Calculate similarity scores between a query embedding and a dictionary of embeddings.
        
        Args:
            query_embedding (numpy.ndarray): The query embedding vector
            embedding_dict (dict): Dictionary with keys as identifiers and values as embeddings
            
        Returns:
            dict: Dictionary of similarity scores for each item in embedding_dict
        """
        similarities = {}
        for key, embedding in embedding_dict.items():
            # Calculate cosine similarity
            similarity = self.calculate_similarity(query_embedding, embedding)
            similarities[key] = similarity
        
        return similarities
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate similarity between two embeddings using cosine similarity.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize the vectors
        embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
        embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        return np.dot(embedding1_normalized, embedding2_normalized)

class OpenAIEmbeddingService(EmbeddingServiceInterface):
    """
    Embedding service using OpenAI API.
    """
    
    def __init__(self, api_key=None, model_name="text-embedding-ada-002"):
        """
        Initialize the OpenAI embedding service.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, it will look for OPENAI_API_KEY env var
            model_name (str): Name of the OpenAI embedding model to use
        """
        try:
            import openai
            if api_key:
                openai.api_key = api_key
            self.client = openai.OpenAI()
            self.model_name = model_name
        except ImportError:
            raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'")
    
    def generate_embedding(self, text):
        """
        Generate a vector embedding for a single text using OpenAI.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return np.array(response.data[0].embedding)
    
    def generate_embeddings(self, texts):
        """
        Generate vector embeddings for multiple texts using OpenAI.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return np.array([data.embedding for data in response.data])

# Default embedding service using SentenceTransformer
embedding_service = OpenAIEmbeddingService()

