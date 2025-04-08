from serverRouter.smartRouter.classifyPrompt import classify_prompt_with_confidence
from serverRouter.smartRouter.model_ranking import aggregate_model_metrics, rank_models
from serverRouter.smartRouter.param_types import LatencyType, CostType
import json

with open('serverRouter/smartRouter/database/_task_models.json', 'r') as f:
    task_models = json.load(f)

def SmartRouter(messages, max_latency, max_cost, model_list):
    """
    Streaming version of SmartRouter that uses confidence-enhanced classification
    
    Args:
        messages (list): List of chat messages
        max_latency (str or float): Maximum latency value (string name or numeric value)
        max_cost (str or float): Maximum cost value (string name or numeric value)
        model_list (list): List of specific models to consider
        
    Yields:
        dict: Progress updates and intermediate results at each step
    """
    # Initial state
    max_latency_enum = LatencyType.from_value(max_latency)
    max_cost_enum = CostType.from_value(max_cost)
    
    # For numeric processing, extract the numeric values if they're enum instances
    latency_value = max_latency_enum.value_in_seconds if isinstance(max_latency_enum, LatencyType) else max_latency
    cost_value = max_cost_enum.value_in_dollars if isinstance(max_cost_enum, CostType) else max_cost

    metadata = {"status": "starting", "message": f"Routing with {max_latency} speed and {max_cost} cost", "data": {"max_latency": max_latency, "max_cost": max_cost}}
    yield {"event": "metadata", "data": json.dumps(metadata)}

    # Get the user query
    user_queries = [msg.content for msg in messages[-2:] if msg.role == "user"]
    query = " ".join(user_queries) if len(user_queries) >= 2 else messages[-1].content

    # Enhanced classification with confidence scores
    classification_result = classify_prompt_with_confidence(query)
    
    # Extract tasks and their similarity scores, excluding metadata
    task_scores = {k: v.get("similarity", 0) for k, v in classification_result.items() if k != "_meta" and isinstance(v, dict)}
    
    # Log detailed classification for debugging
    confidence_info = {k: v.get("confidence", "N/A") for k, v in classification_result.items() if k != "_meta" and isinstance(v, dict)}
    needs_domain_analysis = classification_result.get("_meta", {}).get("needs_domain_analysis", False)
    
    classification_log = {
        "tasks": task_scores,
        "confidence": confidence_info,
        "needs_domain_analysis": needs_domain_analysis
    }
    
    metadata = {"status": "classified", "message": f"Query classified with confidence", "data": classification_log}
    yield {"event": "metadata", "data": json.dumps(metadata)}  # Structured event
    
    # Use extracted task_scores for aggregation
    models_aggregated = aggregate_model_metrics(task_scores, task_models)
    metadata = {"status": "aggregated", "message": "Aggregated model metrics with confidence adjustment", "data": {"models_aggregated": models_aggregated}}
    yield {"event": "metadata", "data": json.dumps(metadata)}  # Structured event
    
    # Check if domain analysis is needed and log strategy
    if needs_domain_analysis:
        metadata = {"status": "strategy", 
                   "message": "Using generalist model strategy due to ambiguous classification",
                   "data": {"strategy": "generalist"}}
        yield {"event": "metadata", "data": json.dumps(metadata)}
    
    # Pass the full classification result (with confidence) and numeric values to ranking
    best_model = rank_models(models_aggregated, latency_value, cost_value, model_list, classification_result, task_models)

    # Check if best_model indicates an error or default state
    if 'model' not in best_model or not best_model.get('model'):
        # Handle cases where rank_models returns a default/error state directly
        metadata = {"status": "error" if "message" in best_model else "ranked", 
                    "message": best_model.get("message", "Ranking failed or no suitable model found"), 
                    "data": best_model}
        yield {"event": "metadata", "data": json.dumps(metadata)}
        yield {"event": "return", "data": json.dumps(best_model)} # Return the error/default state
        return # Stop processing

    # Proceed if a valid model was returned
    metadata = {"status": "ranked", "message": best_model.get("message", f"Ranked models, selected {best_model['model']}")}
    yield {"event": "metadata", "data": json.dumps(metadata)}  # Structured event
    
    # Construct final message based on whether cost/latency are available
    cost_str = f"{best_model.get('cost', 'N/A')} $/k tokens" if best_model.get('cost') is not None else "cost N/A"
    latency_str = f"{best_model.get('latency', 'N/A')} second latency" if best_model.get('latency') is not None else "latency N/A"
    
    final_message = f"Selected {best_model['model']} with score {best_model.get('score', 'N/A')}, {cost_str}, and {latency_str}"
    metadata = {"status": "complete", "message": final_message, "data": {"best_model": best_model}}
    yield {"event": "metadata", "data": json.dumps(metadata)}  # Structured event

    yield {"event": "return", "data": json.dumps(best_model)}

