from serverRouter.smartRouter.classifyPrompt import classify_prompt
from serverRouter.smartRouter.model_ranking import aggregate_model_metrics, rank_models
from serverRouter.smartRouter.param_types import LatencyType, CostType
import json

with open('serverRouter/smartRouter/database/_task_models.json', 'r') as f:
    task_models = json.load(f)

def SmartRouter(messages, max_latency, max_cost, model_list):
    """
    Streaming version of SmartRouter that yields intermediate results at each step.
    
    Args:
        messages (list): List of chat messages
        max_latency (str or float): Maximum latency value (string name or numeric value)
        max_cost (str or float): Maximum cost value (string name or numeric value)
        
    Yields:
        dict: Progress updates and intermediate results at each step
    """
    # Initial state
    max_latency_enum = LatencyType.from_value(max_latency)
    max_cost_enum = CostType.from_value(max_cost)
    
    # For numeric processing, extract the numeric values if they're enum instances
    latency_value = max_latency_enum.value_in_seconds if isinstance(max_latency_enum, LatencyType) else max_latency
    cost_value = max_cost_enum.value_in_dollars if isinstance(max_cost_enum, CostType) else max_cost

    metadata = {"status": "starting", "message": f"Routing to a {max_latency} speed model with {max_cost} cost", "data": {"max_latency": max_latency, "max_cost": max_cost}}
    yield {"event": "metadata", "data": metadata}  # Structured event

    # Get the last two user queries if available
    user_queries = [msg.content for msg in messages[-2:] if msg.role == "user"]
    query = " ".join(user_queries) if len(user_queries) >= 2 else messages[-1].content

    similar_tasks = classify_prompt(query)
    metadata = {"status": "classified", "message": f"Prompt classified: {', '.join([f'{task} ({score})' for task, score in similar_tasks.items()])}", "data": {"similar_tasks": similar_tasks}}
    yield {"event": "metadata", "data": metadata}  # Structured event
    
    models_aggregated = aggregate_model_metrics(similar_tasks, task_models)
    metadata = {"status": "aggregated", "message": "Aggregated model metrics based on prompt, latency, and cost", "data": {"models_aggregated": models_aggregated}}
    yield {"event": "metadata", "data": metadata}  # Structured event
    
    # Pass the numeric values to the ranking function
    best_model = rank_models(models_aggregated, latency_value, cost_value, model_list)

    metadata = {"status": "ranked", "message": f"{best_model['message']}"}
    yield {"event": "metadata", "data": metadata}  # Structured event
    
    metadata = {"status": "complete", "message": f"Selected {best_model['model']} with highest score, {best_model['cost']/1000} $/k tokens, and {best_model['latency']} second latency", "data": {"best_model": best_model}}
    yield {"event": "metadata", "data": metadata}  # Structured event

    yield {"event": "return", "data": best_model}

