from serverRouter.smartRouter.taskEmbeddingManager import task_manager

def classify_prompt(query):
    """
    Classifies a given prompt and returns a dictionary of similar tasks with their normalized similarity scores.
    
    Args:
        query (str): The prompt to classify
        task_manager: The task manager object with find_similar_tasks method
        loaded_embeddings: The embeddings to use for classification
        
    Returns:
        dict: Dictionary mapping task_id to similarity score (only includes scores > 0)
    """
    
    similar_tasks = task_manager.find_similar_tasks(query)
    
    # Normalize similarity scores to 0-1 range
    min_score = min(score for _, score in similar_tasks)
    max_score = max(score for _, score in similar_tasks)
    score_range = max_score - min_score
    
    similar_tasks = [(task_id, (score - min_score) / score_range if (score - min_score) / score_range >= 0.5 else 0) 
                    for task_id, score in similar_tasks]
    
    # Convert to dictionary, only including scores > 0
    result = {}
    for task_id, similarity in similar_tasks:
        if similarity > 0:
            result[task_id] = float(round(similarity, 4))
            
    return result


# Example usage:
if __name__ == "__main__":
    query = "code a python program to calculate the sum of all numbers in a list"
    print(f"\nFinding tasks similar to: '{query}'")
    
    similar_tasks_dict = classify_prompt(query)
    
    print("\nTop similar tasks:")
    for task_id, similarity in similar_tasks_dict.items():
        print(f"- {task_id} (Similarity: {similarity:.4f})")