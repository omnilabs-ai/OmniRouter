import time
from .confidence_utils import calculate_confidence_score, check_if_needs_domain_analysis

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

def classify_prompt_with_confidence(query):
    """
    Enhanced version of classify_prompt that returns both similarity and confidence scores
    """
    similar_tasks_raw = task_manager.find_similar_tasks(query)
    
    if not similar_tasks_raw:
        return {"_meta": {"needs_domain_analysis": True, "query_length": len(query.split()), "timestamp": time.time(), "original_query": query}}

    # Normalize similarity scores to 0-1 range
    scores = [score for _, score in similar_tasks_raw]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # Store normalized scores for confidence calculation
    normalized_similar_tasks = [] 
    for task_id, score in similar_tasks_raw:
        normalized_score = (score - min_score) / score_range if score_range > 0 else 0
        normalized_similar_tasks.append((task_id, normalized_score))

    # Calculate confidence metrics
    result = {}
    for task_id, normalized_score in normalized_similar_tasks:
        if normalized_score >= 0.5:  # Keep original threshold for initial filtering
            # Calculate confidence using the raw normalized scores list
            confidence = calculate_confidence_score(query, task_id, normalized_score, normalized_similar_tasks)
            
            result[task_id] = {
                "similarity": float(round(normalized_score, 4)),
                "confidence": float(round(confidence, 4)),
                "classification_type": "primary" if confidence > 0.8 else "secondary"
            }
    
    # Check if we need domain-specific analysis based on the calculated results
    needs_domain_analysis = check_if_needs_domain_analysis(result)
    
    # Add metadata about the classification
    result["_meta"] = {
        "needs_domain_analysis": needs_domain_analysis,
        "query_length": len(query.split()),
        "timestamp": time.time(),
        "original_query": query
    }
            
    return result

# Example usage:
if __name__ == "__main__":
    query = "code a python program to calculate the sum of all numbers in a list"
    print(f"\nFinding tasks similar to: '{query}'")
    
    similar_tasks_dict = classify_prompt(query)
    
    print("\nTop similar tasks:")
    for task_id, similarity in similar_tasks_dict.items():
        print(f"- {task_id} (Similarity: {similarity:.4f})")