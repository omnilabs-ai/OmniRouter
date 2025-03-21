"""
This script will load in all the data from the database and make all the updates needed.
"""
import json
import numpy as np
from serverRouter.smartRouter.taskEmbeddingManager import task_manager

class TaskModelGenerator:
    def __init__(self, task_benchmarks_path='smartRouter/database/task_benchmarks.json',
                 model_benchmarks_path='smartRouter/database/model_benchmarks.json',
                 output_path='smartRouter/database/_task_models.json'):
        self.task_benchmarks_path = task_benchmarks_path
        self.model_benchmarks_path = model_benchmarks_path
        self.output_path = output_path

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def create_task_vector(self, task_benchmarks, all_benchmarks):
        """Create a normalized vector representation of task benchmark weights"""
        vector = []
        for benchmark in all_benchmarks:
            weight = task_benchmarks.get(benchmark, 0)
            vector.append(weight)
        return vector

    def create_model_vector(self, model_data, all_benchmarks):
        """Create a normalized vector representation of model benchmark scores"""
        vector = []
        for benchmark in all_benchmarks:
            # Skip non-benchmark fields like provider, max_tokens, etc.
            if benchmark in model_data and benchmark in all_benchmarks:
                score = model_data.get(benchmark, 0)
                vector.append(score)
            else:
                vector.append(0)
        return vector

    def get_top_k_models_for_task(self, task_benchmarks, model_benchmarks, k=3):
        """Find the top k models for a task based on similarity score"""
        # Get all unique benchmarks
        all_benchmarks = set()
        for task, benchmarks in task_benchmarks.items():
            all_benchmarks.update(benchmarks.keys())
        all_benchmarks = list(all_benchmarks)
        
        results = {}
        
        for task, task_bench in task_benchmarks.items():
            task_vector = self.create_task_vector(task_bench, all_benchmarks)
            
            # Get model information with their similarity scores
            model_info = []
            for model_name, model_data in model_benchmarks.items():
                model_vector = self.create_model_vector(model_data, all_benchmarks)
                # If we have data for the benchmarks in this task
                if any(benchmark in model_data for benchmark in task_bench):
                    # Convert vectors to numpy arrays
                    task_vector_np = np.array(task_vector)
                    model_vector_np = np.array(model_vector)
                    
                    # Calculate similarity (accuracy)
                    similarity = np.dot(task_vector_np, model_vector_np)
                    
                    # Extract cost and latency information if available
                    cost = model_data.get('tokenCost', float('inf'))
                    latency = model_data.get('latency', float('inf'))
                    
                    model_info.append({
                        'name': model_name,
                        'accuracy': similarity,
                        'cost': cost,
                        'latency': latency
                    })
            
            # Sort by similarity score (descending)
            model_info.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Get top k models
            top_k_models_info = model_info[:k]
            
            # Create task models dictionary without normalization
            task_models = {}
            
            for model in top_k_models_info:
                task_models[model['name']] = {
                    "accuracy": model['accuracy'],
                    "cost": model['cost'],
                    "latency": model['latency']
                }
                
            results[task] = task_models
        
        return results

    def generate(self, k=3):
        # Load the data
        task_benchmarks = self.load_json(self.task_benchmarks_path)
        model_benchmarks = self.load_json(self.model_benchmarks_path)

        # Generate the task to model mapping
        task_models = self.get_top_k_models_for_task(task_benchmarks, model_benchmarks, k=k)
        
        # Save the results
        self.save_json(task_models, self.output_path)
        
        
        return task_models


if __name__ == "__main__":
    generator = TaskModelGenerator()
    generator.generate(k=5)
    print(f"Generated {generator.output_path} successfully")


    json_file_path = "serverRouter/smartRouter/database/task_info.json"
    embeddings_path = "serverRouter/smartRouter/database/_task_embeddings.pkl"
    task_embeddings = task_manager.process_task_json(json_file_path)
    success = task_manager.save_embeddings(embeddings_path)
    print(f"Generated {embeddings_path} successfully")


# python -m smartRouter.updateDB