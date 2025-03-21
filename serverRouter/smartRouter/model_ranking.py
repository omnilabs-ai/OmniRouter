def rank_models(models_aggregated, max_latency, max_cost, model_list):

    # Filter models based on max_latency and max_cost
    filtered_models = {}
    for model_name, model_info in models_aggregated.items():
        if model_info['latency'] <= max_latency and model_info['cost'] <= max_cost:
            filtered_models[model_name] = model_info
    
    # If model_list is provided and not empty, only consider models from the list
    if model_list:
        filtered_list_models = {}
        for model_name, model_info in filtered_models.items():
            if model_name in model_list:
                filtered_list_models[model_name] = model_info
        
        # If there are models in both filtered_models and model_list
        if filtered_list_models:
            filtered_models = filtered_list_models
        # If none of the models in model_list meet criteria, select the best model from model_list
        elif model_list:
            # Select models from model_list that exist in models_aggregated
            list_models = {model: models_aggregated[model] for model in model_list if model in models_aggregated}
            if list_models:
                # Find model with highest accuracy from list_models
                best_list_model = max(list_models.items(), key=lambda x: x[1]['accuracy'])
                return {"model": best_list_model[0], "score": best_list_model[1]['accuracy'], 
                        "cost": best_list_model[1]['cost'], "latency": best_list_model[1]['latency'],
                        "message": "No models meet criteria, selecting best model from given models"}

    # Return empty dict if no models meet criteria
    if not filtered_models:
        return {"gpt-4o-mini": {
            "accuracy": 1,
            "cost": 0.6,
            "latency": 0.56,
            "message": "No models meet criteria, selecting default model"
        }}
    
    # Find model with highest accuracy
    best_model = max(filtered_models.items(), key=lambda x: x[1]['accuracy'])
    
    # Return dict with just the best model
    return {"model": best_model[0], "score": best_model[1]['accuracy'], "cost": best_model[1]['cost'], "latency": best_model[1]['latency'], "message": "Optimal model found"}

def aggregate_model_metrics(similar_tasks, task_models):
    """
    Aggregates model metrics across similar tasks, adjusting accuracy by similarity.
    
    Args:
        similar_tasks (dict): Dictionary of task IDs with their similarity scores
        task_models (dict): Dictionary of tasks with their model metrics
        
    Returns:
        dict: Aggregated model metrics
    """
    models_aggregated = {}
    
    for task_id, similarity in similar_tasks.items():
        if task_id in task_models:
            for model_name, model_info in task_models[task_id].items():
                # Adjust accuracy by task similarity
                adjusted_accuracy = model_info['accuracy'] * similarity
                
                # If model already exists, add the adjusted accuracy
                if model_name in models_aggregated:
                    models_aggregated[model_name]['accuracy'] += adjusted_accuracy
                    # Use the best (lowest) cost and latency values if this instance is better
                    models_aggregated[model_name]['cost'] = min(models_aggregated[model_name]['cost'], model_info['cost'])
                    models_aggregated[model_name]['latency'] = min(models_aggregated[model_name]['latency'], model_info['latency'])
                else:
                    # First time seeing this model
                    models_aggregated[model_name] = {
                        'accuracy': adjusted_accuracy,
                        'cost': model_info['cost'],
                        'latency': model_info['latency']
                    }
    
    return models_aggregated 