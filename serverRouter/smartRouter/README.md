# SmartRouter

The SmartRouter is an intelligent AI model router system that selects the best AI model for a given user query. It uses a combination of task classification and model benchmarks to make optimal model selections.

## Key Components

### 1. Smart Router Core (`smart_router.py`)

The main router component that:
- Identifies the task types in a user query
- Scores available models based on their benchmark performance for identified tasks
- Selects the optimal model based on user preferences (cost, latency, accuracy)
- Dynamically loads provider information from the model registry
- Raises explicit errors when the model registry is unavailable

### 2. Task Vector Database (`task_vector_db.py`)

A semantic task classification system that:
- Stores examples of different types of tasks (coding, math, creative writing, etc.)
- Uses embeddings to calculate semantic similarity between user queries and task examples
- Provides confidence scores for different task categories

### 3. Embedding Model (`embedding_model.py`)

Provides vector embeddings for text using OpenAI's API:
- Generates embeddings for queries and task examples
- Caches embeddings to improve performance
- Calculates similarity between embeddings

### 4. Session Tracker (`session_tracker.py`)

Tracks user sessions and interactions:
- Monitors user engagement with responses
- Provides data for model selection

### 5. Configuration Manager (`config.py`)

Centralized configuration system that:
- Provides default settings for all components
- Enables environment variable overrides
- Includes settings for caching and provider adjustments

## Utility Scripts

### 1. Task Database Generator (`generate_task_db.py`)

Creates a database of task examples:
- Initializes the vector database with examples
- Computes embeddings for the examples
- Saves the database for runtime use

### 2. Benchmark Embedding Generator (`benchmark_embedding_generator.py`)

Creates vector embeddings for benchmarks:
- Maps benchmarks (MMLU, HumanEval, etc.) to their descriptions
- Generates embeddings for benchmark descriptions
- Enables finding the most relevant benchmarks for a query

## How It Works

1. When a user sends a query, the SmartRouter identifies the tasks involved using:
   - Semantic similarity to task examples using the task vector database
   - Keyword matching for specific task types
   - Pattern matching for certain query types

2. The router then scores available models based on:
   - Their benchmark performance on relevant tasks
   - User preferences (cost vs. latency vs. accuracy)
   - Provider-specific adjustments from the configuration

3. The router selects the top models and:
   - Ensures provider diversity
   - Records the selection for tracking

## Model Registry Integration

The SmartRouter integrates with the model registry system to:

1. **Dynamically load model information**: Model metadata, benchmarks, and costs are directly loaded from the model registry
2. **Auto-update when new models are added**: No manual updates needed when new models are added to the registry
3. **Explicit error handling**: Clear error messages when the model registry is unavailable
4. **Provider information**: Provider display names are automatically loaded from ModelProvider enum

This integration ensures that:
- The router stays in sync with available models
- No hard-coded model information is needed
- The system fails explicitly when dependencies are unavailable

## Testing

The SmartRouter components can be tested using the unit tests in `testLib/test_smart_router.py`.

## Setup and Configuration

1. The SmartRouter requires:
   - OpenAI API key for embeddings
   - Task vector database (can be generated)
   - Benchmark embeddings (can be generated)
   - Access to the model registry

2. The router loads configuration from environment variables and configuration files.

3. Cached data is stored in the `cache/` directory to improve performance.

## Core Capabilities

- **Intelligent Task Identification**: Automatically detects tasks (coding, math, creative writing, etc.) from user queries
- **Model Performance Matching**: Routes queries to models with proven performance on similar tasks
- **Personalized Routing**: Adapts to user preferences for cost, latency, and accuracy
- **Provider Diversity**: Ensures model selections come from diverse providers for robustness
- **Dynamic Updates**: Automatically adapts to changes in the model registry

## Usage Examples

### Basic Model Selection

```python
from serverRouter.smartRouter import SmartRouter

# Initialize the router
router = SmartRouter()

# Select models for a query
request = SmartRouterRequest(
    messages=[ChatMessage(role="user", content="Explain quantum computing in simple terms")],
    k=3
)
result = router.select_models(request)

print(f"Selected models: {result['selected_models']}")
```

### Model Selection with Custom Preferences

```python
# Create request with custom preference weights (cost: 0.2, latency: 0.3, accuracy: 0.5)
request = SmartRouterRequest(
    messages=[ChatMessage(role="user", content="Explain quantum computing in simple terms")],
    k=3,
    rel_cost=0.2,
    rel_latency=0.3,
    rel_accuracy=0.5
)

# Select models with custom preferences
result = router.select_models(request)
```

## Generating Benchmark Embeddings

The SmartRouter needs benchmark embeddings to match user queries with appropriate models based on benchmark performance. To generate these embeddings:

```bash
python -m serverRouter.smartRouter.benchmark_embedding_generator
```

This will:
1. Create embeddings for each benchmark in the BENCHMARKS dictionary
2. Save the embeddings to `serverRouter/smartRouter/benchmark_embeddings.pkl`

### Command-line Options

```bash
python -m serverRouter.smartRouter.benchmark_embedding_generator --help
```

Available options:
- `--api_key`: Specify OpenAI API key (optional, will use environment variable if not provided)
- `--output`: Path to save embeddings pickle file (default: serverRouter/smartRouter/benchmark_embeddings.pkl)
- `--cache_dir`: Directory to cache embeddings (default: cache)
- `--env_file`: Path to .env file containing API keys (default: .env)

## Task Vector Database

The task vector database helps with classifying user queries into different task types. To generate the task database:

```bash
python -m serverRouter.smartRouter.generate_task_db
```

This will create a vector database with default task examples and save it to `serverRouter/smartRouter/task_examples_db.pkl`.

You can also customize the task examples by editing the `task_examples.json` file before running the generator.

## Provider Diversity

SmartRouter ensures model selections include a diverse set of providers, which helps with:

1. **Robustness**: Minimizing impact if a specific provider has issues
2. **Specialized Capabilities**: Leveraging the unique strengths of different providers
3. **Cost Management**: Balancing performance with cost across providers

You can configure provider diversity settings in the `config.py` file.

## File Organization

The SmartRouter directory contains these important files:

- `smart_router.py`: Main router implementation
- `config.py`: Configuration settings
- `task_vector_db.py`: Vector database for task classification
- `embedding_model.py`: Handles embeddings for semantic similarity
- `session_tracker.py`: Tracks user sessions for context-aware routing
- `benchmark_embedding_generator.py`: Generates benchmark embeddings
- `generate_task_db.py`: Creates the task vector database
- `benchmark_embeddings.pkl`: Pre-computed embeddings for benchmarks

## Advanced Configuration

The SmartRouter behavior can be configured through the `config.py` file, which includes settings for:
- File paths and resource locations
- Task identification methods and thresholds
- Provider diversity and adjustments
- Caching behavior and performance tuning

## Extending SmartRouter

SmartRouter can be extended in several ways:

1. **Custom Task Patterns**: Add patterns to the TASK_KEYWORDS dictionary
2. **New Task Types**: Add new task types to the task database
3. **Custom Benchmarks**: Define new benchmarks in the TASK_TO_BENCHMARK mapping
4. **Provider Adjustments**: Add provider-specific adjustments in the config

## Troubleshooting

Common issues and solutions:

1. **Missing Embeddings**: If benchmark embeddings are missing, run the benchmark_embedding_generator
2. **Task Identification Issues**: Improve task examples in the task database and regenerate
3. **Model Diversity**: Check provider diversity settings if too many models from the same provider are being selected
4. **Model Registry Errors**: Ensure the model registry is properly configured and accessible
