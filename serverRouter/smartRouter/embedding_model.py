from openai import OpenAI
import numpy as np

class OpenAIEmbeddings:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings client.
        
        Args:
            api_key (str | None): OpenAI API key. If None, will try to use OPENAI_API_KEY env variable
            model (str): OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a vector embedding."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Ensure the embeddings are normalized
        embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
        embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1_normalized, embedding2_normalized))
