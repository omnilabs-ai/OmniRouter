# Smart Router Documentation

## Overview

The Smart Router is a key component of the OmniRouter system that intelligently routes user queries to the most appropriate AI model based on a variety of factors, including:

1. **Task Classification**: Identifying the type of task in the user's query (coding, math, creative writing, etc.)
2. **Model Performance**: Evaluating which models perform best on relevant benchmarks
3. **User Preferences**: Balancing performance, cost, and speed based on user-defined preferences
4. **Context Analysis**: Analyzing the specific needs in the user's prompt

## Key Components

### Task Identification System

The Smart Router uses keyword matching and semantic analysis to classify user queries into different task categories:

- **Coding**: Programming-related queries, code generation, debugging
- **Math**: Mathematical calculations, equations, statistics
- **Science**: Scientific queries across physics, chemistry, biology, etc.
- **Reasoning**: Logical problems, deductive reasoning, analysis
- **General Knowledge**: Factual inquiries, explanations, definitions
- **Creative Writing**: Story generation, content creation, creative tasks

Each category has associated keywords and patterns that help identify the nature of the request.

### Benchmark Mapping

Different tasks correlate with different benchmarks that measure model performance:

- **HumanEval**: Code generation capability
- **MMLU**: General knowledge and reasoning
- **MATH**: Advanced mathematical reasoning
- **MGSM**: Multi-step math problem solving
- **BFCL**: Logical reasoning and chain-of-thought
- **GPQA**: Graduate-level physics and scientific reasoning

The Smart Router maps identified tasks to relevant benchmarks with appropriate weights.

### Model Scoring

Models are scored based on:

1. **Benchmark Performance**: How well the model performs on benchmarks relevant to the identified task
2. **Cost Efficiency**: The cost per token for the model
3. **Latency**: How quickly the model responds
4. **Context Window**: Maximum tokens the model can process (when relevant)

User preferences determine the weight of each factor in the final score.

## Usage

### Basic Usage

The simplest way to use the Smart Router is through the `/v1/router/select-model` endpoint:

```python
response = requests.post(
    "https://api.omnirouter.ai/v1/router/select-model",
    json={
        "messages": [{"role": "user", "content": "Write a function to calculate prime numbers"}]
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

This will:
1. Analyze the query
2. Select the best model for this task using default preferences
3. Send the query to the selected model
4. Return the model's response

### Advanced Usage with Custom Preferences

```python
response = requests.post(
    "https://api.omnirouter.ai/v1/router/select-model",
    json={
        "messages": [{"role": "user", "content": "Write a function to calculate prime numbers"}],
        "rel_cost": 0.7,          # High importance on cost efficiency
        "rel_latency": 0.1,       # Low importance on speed
        "rel_accuracy": 0.2,      # Moderate importance on accuracy
        "k": 3,                   # Get top 3 model recommendations
        "verbose": True,          # Include explanation of model selection
        "model_names": ["gpt-4", "claude-3-opus"]  # Optional restriction to specific models
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

### Streaming Responses

For streaming responses, use the `/v1/router/select-model-stream` endpoint:

```python
response = requests.post(
    "https://api.omnirouter.ai/v1/router/select-model-stream",
    json={
        "messages": [{"role": "user", "content": "Write a poem about AI"}],
        "rel_accuracy": 0.8,      # Prioritize quality for creative writing
    },
    headers={"Authorization": f"Bearer {api_key}"},
    stream=True
)

for chunk in response.iter_lines():
    if chunk:
        # Process SSE format chunks
        data = chunk.decode('utf-8').lstrip('data: ')
        if data != '[DONE]':
            try:
                json_data = json.loads(data)
                print(json_data.get('content', ''), end='', flush=True)
            except json.JSONDecodeError:
                pass
```

## How It Works: Technical Details

### Embedding-Based Semantic Matching

The Smart Router uses vector embeddings to:

1. Capture semantic meaning of user queries beyond keywords
2. Match queries to benchmark categories
3. Handle edge cases where keywords may be misleading

### Weighted Scoring Algorithm

The formula for scoring models is:

```
final_score = (accuracy_score * rel_accuracy) +
              (cost_score * rel_cost) +
              (latency_score * rel_latency)
```

Where:
- `accuracy_score` is weighted by benchmark relevance
- `cost_score` is normalized across available models
- `latency_score` is normalized across available models
- `rel_accuracy`, `rel_cost`, and `rel_latency` are user preference weights

### Task-Benchmark Mapping Matrix

Tasks are mapped to benchmarks with the following weights:

| Task              | HumanEval | MMLU | MATH | MGSM | BFCL | GPQA |
|-------------------|-----------|------|------|------|------|------|
| Coding            | 0.7       | 0.1  | 0.0  | 0.0  | 0.2  | 0.0  |
| Math              | 0.0       | 0.1  | 0.6  | 0.3  | 0.0  | 0.0  |
| Science           | 0.0       | 0.3  | 0.1  | 0.0  | 0.0  | 0.6  |
| Reasoning         | 0.0       | 0.3  | 0.0  | 0.2  | 0.5  | 0.0  |
| General Knowledge | 0.0       | 0.8  | 0.0  | 0.0  | 0.1  | 0.1  |
| Creative Writing  | 0.2       | 0.5  | 0.0  | 0.0  | 0.3  | 0.0  |

## Integration Examples

### Python SDK Example

```python
from omnirouter import OmniRouter

router = OmniRouter(api_key="your_api_key")

# Get model recommendation
model_info = router.select_model(
    query="Implement a binary search tree in Python",
    rel_cost=0.3,
    rel_accuracy=0.7
)

print(f"Selected model: {model_info['model']}")
print(f"Task identified: {model_info['identified_tasks']}")

# Send directly to selected model
response = router.complete(
    messages=[{"role": "user", "content": "Implement a binary search tree in Python"}],
    # Preferences automatically applied from previous selection
)

print(response.content)
```

### Function Calling Example

```python
# The Smart Router handles function calling by selecting models that support it
response = router.complete(
    messages=[{"role": "user", "content": "What's the weather in New York?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)

# Smart Router will select a model that supports function calling
```

## Best Practices

1. **Provide Clear Queries**: While the Smart Router analyzes queries intelligently, clearer queries lead to better model selection
2. **Set Appropriate Preferences**: Tune rel_cost, rel_latency, and rel_accuracy based on your actual priorities
3. **Consider Model Restrictions**: Use model_names to restrict selection to models you've tested and approved
4. **Review the Explanation**: When verbose=True, review the explanation to understand why models were selected

## Performance Considerations

- **Cold Start Latency**: First request may have higher latency due to model loading
- **Benchmark Updates**: Model performance benchmarks are updated periodically
- **Caching**: The Smart Router caches embedding calculations to improve performance on similar queries
- **Analysis Overhead**: Task identification adds minimal overhead (typically <50ms)
- **Streaming Performance**: When using streaming, model selection happens once at the beginning

## Customization and Extension

### Adding New Models

When adding new models to the system, provide benchmark scores to ensure proper routing:

```python
from serverRouter.core.datamodels import ModelInfo, ModelProvider, BenchmarkScores

new_model = ModelInfo(
    name="new-model-v1",
    provider=ModelProvider.CUSTOM,
    description="Description of the new model's capabilities",
    max_tokens=32768,
    benchmarks={
        "MMLU": 0.85,
        "GPQA": 0.45,
        "HumanEval": 0.78,
        "MATH": 0.62,
        "BFCL": 0.79,
        "MGSM": 0.81
    },
    tokenCost=12.0,  # Cost per million tokens
    latency=0.52     # Response time in seconds
)

# Register model with Smart Router
CHAT_MODELS["new-model"] = new_model
```

### Customizing Task Detection

You can customize the task detection by modifying the `TASK_KEYWORDS` dictionary:

```python
from serverRouter.smartRouter.smart_router import TASK_KEYWORDS

# Add new keywords for existing tasks
TASK_KEYWORDS["coding"].extend(["golang", "rust", "docker"])

# Create new task categories
TASK_KEYWORDS["medical"] = [
    "diagnosis", "treatment", "symptoms", "disease", "patient",
    "medical", "clinical", "healthcare", "doctor", "nurse"
]

# Update task-to-benchmark mapping
TASK_TO_BENCHMARK_WEIGHTS["medical"] = {
    "MMLU": 0.7,
    "GPQA": 0.3
}
```

## Monitoring and Diagnostics

### Performance Logging

The Smart Router logs detailed information about its decisions:

```
INFO:smart_router:Task identified: {'coding': 0.72, 'math': 0.12, 'general_knowledge': 0.16}
INFO:smart_router:Selected model: gpt-4 (Score: 0.876)
INFO:smart_router:Key benchmarks: HumanEval (0.866), BFCL (0.883)
```

These logs can be used to monitor and improve routing performance.

### Debugging Model Selection

To debug model selection, use the `verbose=True` parameter:

```python
response = router.select_model(
    query="Complex query here",
    verbose=True
)

print(response["explanation"])
```

This will provide a detailed explanation of why specific models were selected, including:
- Task analysis
- Benchmark relevance
- Model scores
- Preference weighting

## Error Handling

The Smart Router handles several error cases gracefully:

1. **No Suitable Models**: If no models meet the criteria, falls back to a default model
2. **Model Unavailability**: If a selected model is unavailable, tries the next best model
3. **Benchmark Data Missing**: If benchmark data is missing, uses available data and logs warnings
4. **Invalid Preferences**: If preference weights don't sum to 1.0, automatically normalizes them

## Security and Privacy

- The Smart Router does not store user queries beyond the current request
- Embedding calculations are performed locally without sending data to external services
- Model selection operates on the semantic meaning of queries, not specific details
- No data from queries is used to update or train the Smart Router itself

## Future Enhancements

Planned enhancements to the Smart Router include:

1. **Multi-modal Routing**: Support for routing requests with images, audio, or other media
2. **Domain-Specific Routing**: Enhanced routing for specialized domains (legal, medical, etc.)
3. **Conversation Context**: Using conversation history to improve model selection
4. **Dynamic Adaptation**: Learning from user feedback to improve future routing decisions
5. **Cost Budgeting**: Support for setting cost limits and optimizing within those constraints

## Common Issues and Troubleshooting

### Issue: Model seems inappropriate for the task

**Solution**: Try one of the following:
- Make the query more specific and clear about the task
- Use `verbose=True` to understand why the model was selected
- Override with specific `model_names` if you have a better model in mind

### Issue: High latency in model selection

**Solution**:
- Check if you're making the first request (cold start)
- Consider simpler preferences (e.g., prioritize just one factor)
- Ensure benchmark data is available for all models

### Issue: Inconsistent model selection for similar queries

**Solution**:
- Use more consistent phrasing in your queries
- If needed, explicitly state the task type in your query
- Consider setting `model_names` to restrict options
