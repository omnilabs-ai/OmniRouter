# Smart Router System

## Overview

The **Smart Router System** dynamically selects the best LLM (Large Language Model) based on query analysis, benchmark performance, and user preferences. It optimizes for accuracy, cost, and latency.

## Core Workflow

### 1. Analyze Query via Embeddings

- When a query arrives, `SmartRouter.get_top_user_models()` converts it into a vector embedding.
- The system compares this embedding against pre-stored benchmark embeddings.
- Each benchmark (e.g., MMLU, GPQA, HumanEval) receives a similarity score to measure relevance to the query.

### 2. Find Best Models for the Task

- The router weights benchmarks based on similarity scores.
- It evaluates each model's performance on these relevant benchmarks.
- A weighted accuracy score is calculated for each model.

### 3. Apply User Preferences

- The system factors in cost and latency according to user-defined importance.
- Models receive normalized scores for accuracy, cost, and latency.
- A final score is computed using user-defined weights.
- The model with the highest weighted score is selected.

### 4. Return Result

- The system returns the best model name.
- If verbose mode is enabled, it provides detailed reasoning for the selection.

## Key Files and Roles

### `benchmarkDB.py`

- Defines benchmarks and their descriptions.
- Generates and stores vector embeddings for benchmarks.

### `embedding_model.py`

- Provides embedding functionality using OpenAI's embedding API.
- Handles text-to-vector conversion and similarity calculations.

### `SmartRouter.py`

- Implements the core logic for model selection.
- Loads benchmark embeddings.
- Computes similarities between queries and benchmarks.
- Evaluates models based on multiple criteria.
- Implements the weighted decision process.

### `datamodels.py`

- Defines the data structures for the system.
- Contains `SmartRouterRequest` for input parameters.
- Contains `ModelInfo` and `BenchmarkScores` for model information.

## Flowchart

```mermaid
graph TD;
    A[User Query] -->|Convert to Vector Embedding| B[Compare with Benchmark Embeddings];
    B -->|Compute Similarity Scores| C[Determine Relevant Benchmarks];
    C -->|Evaluate Model Performance| D[Calculate Weighted Accuracy Scores];
    D -->|Apply User Preferences (Cost, Latency, Accuracy)| E[Compute Final Model Score];
    E -->|Select Best Model| F[Return Model Name & Reasoning];
```

