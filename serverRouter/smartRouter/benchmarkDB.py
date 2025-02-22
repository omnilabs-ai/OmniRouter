"""
IMPORTANT:
    - This file is used to generate the benchmark embeddings for the SmartRouter.
    - The embeddings are saved to the file "benchmark_embeddings.pkl".
    - The embeddings are used to find the most similar benchmark to the query text.
    - The embeddings are used to route the query to the most similar benchmark.

    RUN THIS FILE EVERY TIME YOU ADD A NEW BENCHMARK TO THE BENCHMARKS DICTIONARY.
"""

from typing import Dict
import pickle
from embedding_model import OpenAIEmbeddings

BENCHMARKS: Dict[str, Dict[str, str]] = {
    "MMLU": {
        "name": "Massive Multitask Language Understanding",
        "description": "A comprehensive benchmark covering 57 diverse tasks across subjects like math, history, law, medicine, and more.",
        "purpose": "Tests general knowledge and problem-solving abilities across a broad range of domains.",
        "format": "Multiple-choice questions",
        "significance": "Considered a gold standard for assessing broad knowledge and reasoning in LLMs."
    },
    "GPQA": {
        "name": "Graduate-level Physics Questions and Answers",
        "description": "A benchmark that focuses on advanced physics problems, typically at the graduate level.",
        "purpose": "Tests complex reasoning, mathematical proficiency, and domain-specific scientific knowledge.",
        "format": "Open-ended physics problems",
        "significance": "Evaluates how well an LLM can solve challenging, technical problems, especially in STEM fields."
    },
    "HumanEval": {
        "name": "HumanEval",
        "description": "A benchmark created by OpenAI that evaluates code generation capabilities.",
        "purpose": "Tests an LLM's ability to write correct Python functions based on natural language descriptions.",
        "format": "Python programming problems with unit tests to verify correctness",
        "significance": "Widely used to measure functional correctness in code synthesis."
    },
    "MGSM": {
        "name": "Multi-step Grade School Math",
        "description": "A math benchmark that involves multi-step grade school-level arithmetic and reasoning problems.",
        "purpose": "Evaluates an LLM's multi-step reasoning ability and mathematical problem-solving skills.",
        "format": "Multi-step math problems",
        "significance": "Focuses on whether models can follow logical steps over multiple turns to arrive at correct solutions."
    },
    "BFCL": {
        "name": "BigBench Faithful Chain-of-Thought Logical Reasoning",
        "description": "Part of the BIG-Bench benchmark suite, this subset focuses on faithful chain-of-thought (CoT) logical reasoning.",
        "purpose": "Tests whether an LLM can generate step-by-step reasoning that is both correct and logically valid.",
        "format": "Logical reasoning problems",
        "significance": "Measures how well a model can reason transparently through a problem rather than just guessing the answer."
    },
    "MATH": {
        "name": "Mathematics Aptitude and Training Hardness",
        "description": "A benchmark of high school-level and competition-level math problems.",
        "purpose": "Assesses advanced mathematical reasoning and the LLM's ability to solve difficult math problems with detailed solutions.",
        "format": "Advanced mathematics problems",
        "significance": "Evaluates symbolic reasoning, problem decomposition, and the model's capacity for multi-step problem-solving in math."
    }
}

def generate_benchmark_embeddings(output_file: str = "serverRouter/smartRouter/benchmark_embeddings.pkl", api_key: str | None = None):
    """Generate and save vector embeddings for all benchmark descriptions."""
    model = OpenAIEmbeddings(api_key=api_key)
    
    # Create combined descriptions for each benchmark
    benchmark_texts = {}
    for benchmark_id, info in BENCHMARKS.items():
        combined_text = f"{info['name']}. {info['description']} {info['purpose']} {info['significance']}"
        benchmark_texts[benchmark_id] = combined_text
    
    # Generate embeddings
    embeddings = {}
    for benchmark_id, text in benchmark_texts.items():
        embedding = model.encode(text)
        embeddings[benchmark_id] = embedding
    
    # Save embeddings to file
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

if __name__ == "__main__":
    # You can set your API key here or use environment variable OPENAI_API_KEY
    generate_benchmark_embeddings()

