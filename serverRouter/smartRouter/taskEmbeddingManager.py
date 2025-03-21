import json
import pickle
import os
from serverRouter.smartRouter.embeddingService import embedding_service

class TaskEmbeddingManager:
    def __init__(self, embedding_service):
        """
        Initialize the TaskEmbeddingManager with an embedding service.
        
        Args:
            embedding_service (EmbeddingService, optional): The embedding service to use.
                If None, a new service will be created.
        """
        self.embedding_service = embedding_service
        self.task_embeddings = {}
        
    def process_task_json(self, json_file_path):
        """
        Process a task_info.json file and convert tasks to embeddings.
        
        Args:
            json_file_path (str): Path to the task_info.json file
            
        Returns:
            dict: Dictionary of task embeddings with task IDs as keys
        """
        # Load tasks from JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        # Create a dictionary to store task info and embeddings
        self.task_embeddings = {}
        
        # Process each task
        for task_id, task_data in tasks.items():
            # Combine task description and other relevant fields for embedding
            task_text = self._create_task_text(task_data)
            
            # Generate embedding for the task
            embedding = self.embedding_service.generate_embedding(task_text)
            
            # Store both the original task data and its embedding
            self.task_embeddings[task_id] = {
                'data': task_data,
                'embedding': embedding
            }
        
        return self.task_embeddings
    
    def _create_task_text(self, task_data):
        """
        Create a text representation of a task for embedding.
        
        Args:
            task_data (dict): Task data from the JSON file
            
        Returns:
            str: Text representation of the task
        """
        # Combine relevant fields for better semantic representation
        text_parts = []
        
        if 'title' in task_data:
            text_parts.append(task_data['title'])
        
        if 'description' in task_data:
            text_parts.append(task_data['description'])
        
        if 'keywords' in task_data and isinstance(task_data['keywords'], list):
            text_parts.append(' '.join(task_data['keywords']))
        
        # Join all parts with spaces
        return ' '.join(text_parts)
    
    def save_embeddings(self, output_path):
        """
        Save the task embeddings dictionary to a pickle file.
        
        Args:
            output_path (str): Path to save the pickle file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.task_embeddings, f)
            return True
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            return False
    
    @staticmethod
    def load_embeddings(pickle_path):
        """
        Load task embeddings from a pickle file.
        
        Args:
            pickle_path (str): Path to the pickle file
            
        Returns:
            dict: Dictionary of task embeddings
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Embedding file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            task_embeddings = pickle.load(f)
        
        return task_embeddings
    
    def find_similar_tasks(self, query_text, top_n=6):
        """
        Find tasks similar to a query text.
        
        Args:
            query_text (str): Query text to find similar tasks for
            top_n (int): Number of top results to return
            
        Returns:
            list: List of (task_id, similarity_score) tuples
        """
        if not self.task_embeddings:
            raise ValueError("No task embeddings available. Process tasks first.")
        
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query_text)
        
        # Calculate similarities with all task embeddings
        similarities = {}
        for task_id, task_info in self.task_embeddings.items():
            similarity = self.embedding_service.calculate_similarity(
                query_embedding, 
                task_info['embedding']
            )
            similarities[task_id] = similarity
        
        # Sort by similarity score (descending)
        sorted_similarities = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N results
        return sorted_similarities[:top_n]
    
task_manager = TaskEmbeddingManager(embedding_service)
loaded_embeddings = task_manager.load_embeddings("serverRouter/smartRouter/database/_task_embeddings.pkl")
task_manager.task_embeddings = loaded_embeddings
