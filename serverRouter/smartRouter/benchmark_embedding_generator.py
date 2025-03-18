"""
Benchmark Embeddings Generator

This script creates vector embeddings for benchmarks to enable smart routing based on 
prompt similarity to benchmark categories. It generates embeddings for each benchmark
and saves them to a pickle file for later use by the SmartRouter.

Usage:
    python -m serverRouter.smartRouter.benchmark_embedding_generator
"""

import argparse
import os
import pickle
from typing import Dict, Optional
from pathlib import Path
import logging
import sys
from dotenv import load_dotenv

# Import the embeddings client
from serverRouter.smartRouter.embedding_model import OpenAIEmbeddings

# Define benchmark descriptions with rich metadata
BENCHMARKS = {
    "MMLU": {
        "name": "Massive Multitask Language Understanding",
        "description": "A comprehensive benchmark covering 57 diverse tasks across subjects like math, history, law, medicine, and more.",
        "purpose": "Tests general knowledge and problem-solving abilities across a broad range of domains.",
        "format": "Multiple-choice questions",
        "significance": "Considered a gold standard for assessing broad knowledge and reasoning in LLMs.",
        "examples": [
            "Define the concept of 'stare decisis' in legal contexts and explain its importance.",
            "Identify which of the following compounds would have the highest boiling point and explain why.",
            "Calculate the derivate of f(x) = sin(x^2) with respect to x."
        ]
    },
    "GPQA": {
        "name": "Graduate-level Physics Questions and Answers",
        "description": "A benchmark that focuses on advanced physics problems, typically at the graduate level.",
        "purpose": "Tests complex reasoning, mathematical proficiency, and domain-specific scientific knowledge.",
        "format": "Open-ended physics problems",
        "significance": "Evaluates how well an LLM can solve challenging, technical problems, especially in STEM fields.",
        "keywords": [
            "physics", "science", "graduate level", "technical", "scientific", "stem", "engineering",
            "quantum mechanics", "relativity", "thermodynamics", "mathematics", "equations",
            "formulas", "calculations", "complex problems", "technical expertise", "domain knowledge"
        ],
        "examples": [
            "Derive the Schrödinger equation for a particle in a one-dimensional box.",
            "Calculate the energy levels of a hydrogen atom using the Bohr model.",
            "Explain the principle of general relativity and its implications for spacetime."
        ]
    },
    "HumanEval": {
        "name": "HumanEval",
        "description": "A benchmark created by OpenAI that evaluates code generation capabilities.",
        "purpose": "Tests an LLM's ability to write correct Python functions based on natural language descriptions.",
        "format": "Python programming problems with unit tests to verify correctness",
        "significance": "Widely used to measure functional correctness in code synthesis.",
        "keywords": [
            "coding", "programming", "software development", "algorithms", "python", "code generation",
            "functions", "methods", "implementation", "software engineering", "computer science",
            "debugging", "testing", "programming languages", "development", "coding challenges"
        ],
        "examples": [
            "Write a function to find the nth Fibonacci number.",
            "Implement a binary search algorithm in Python.",
            "Create a function that checks if a string is a palindrome.",
            "Write a sorting algorithm to sort an array of integers."
        ]
    },
    "MGSM": {
        "name": "Multi-step Grade School Math",
        "description": "A math benchmark that involves multi-step grade school-level arithmetic and reasoning problems.",
        "purpose": "Evaluates an LLM's multi-step reasoning ability and mathematical problem-solving skills.",
        "format": "Multi-step math problems",
        "significance": "Focuses on whether models can follow logical steps over multiple turns to arrive at correct solutions.",
        "keywords": [
            "math", "mathematics", "arithmetic", "calculation", "algebra", "geometry", "step-by-step",
            "multi-step", "reasoning", "equations", "elementary math", "grade school", "problem solving",
            "quantitative reasoning", "mathematical operations", "numbers", "formulas"
        ],
        "examples": [
            "If John has 5 apples and Mary gives him 3 more, how many apples does John have?",
            "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
            "What is the area of a rectangle with length 8 cm and width 5 cm?"
        ]
    },
    "BFCL": {
        "name": "BigBench Faithful Chain-of-Thought Logical Reasoning",
        "description": "Part of the BIG-Bench benchmark suite, this subset focuses on faithful chain-of-thought (CoT) logical reasoning.",
        "purpose": "Tests whether an LLM can generate step-by-step reasoning that is both correct and logically valid.",
        "format": "Logical reasoning problems",
        "significance": "Measures how well a model can reason transparently through a problem rather than just guessing the answer.",
        "keywords": [
            "logic", "reasoning", "chain-of-thought", "deduction", "inference", "step-by-step",
            "explanations", "logical thinking", "critical thinking", "problem solving", "analysis",
            "arguments", "premises", "conclusions", "cognitive reasoning", "analytical thinking"
        ],
        "examples": [
            "If all A are B, and all B are C, what can we conclude about A and C?",
            "Given that P implies Q, and Q implies R, what can we say if P is true?",
            "If John is taller than Mary, and Mary is taller than Sue, who is the tallest person?"
        ]
    },
    "MATH": {
        "name": "Mathematics Aptitude and Training Hardness",
        "description": "A benchmark of high school-level and competition-level math problems.",
        "purpose": "Assesses advanced mathematical reasoning and the LLM's ability to solve difficult math problems with detailed solutions.",
        "format": "Advanced mathematics problems",
        "significance": "Evaluates symbolic reasoning, problem decomposition, and the model's capacity for multi-step problem-solving in math.",
        "keywords": [
            "math", "mathematics", "advanced math", "calculus", "algebra", "geometry", "trigonometry",
            "statistics", "probability", "mathematical reasoning", "equations", "proofs", "theorems",
            "competition math", "olympiad", "high school math", "college math", "symbolic reasoning"
        ],
        "examples": [
            "Find all values of x that satisfy the equation x³ - 6x² + 11x - 6 = 0.",
            "Prove that for any positive integer n, the number n³ + 2n is divisible by 3.",
            "Calculate the derivative of f(x) = ln(x²+1) with respect to x."
        ]
    }
}

def generate_benchmark_embeddings(output_file: str = "benchmark_embeddings.pkl", api_key: Optional[str] = None, cache_dir: Optional[str] = None):
    """
    Generate and save vector embeddings for all benchmark descriptions.
    
    Args:
        output_file: Path to save the embeddings pickle file
        api_key: OpenAI API key
        cache_dir: Directory to cache embeddings between runs
    """
    # Create the embeddings client
    model = OpenAIEmbeddings(api_key=api_key, cache_dir=cache_dir)
    
    # Create comprehensive descriptions for each benchmark
    benchmark_texts = {}
    for benchmark_id, info in BENCHMARKS.items():
        # Make sure 'keywords' and 'examples' exist, even if empty
        keywords = info.get('keywords', [])
        examples = info.get('examples', [])
        
        # Combine various aspects of the benchmark for a rich description
        combined_text = (
            f"{info['name']}. {info['description']} {info['purpose']} {info['format']} {info['significance']} "
            f"Keywords: {', '.join(keywords)}. "
            f"Examples: {' '.join(examples)}"
        )
        benchmark_texts[benchmark_id] = combined_text
    
    # Generate embeddings for each benchmark
    embeddings = {}
    for benchmark_id, text in benchmark_texts.items():
        embedding = model.encode(text)
        embeddings[benchmark_id] = embedding
    
    # Create directory for output file if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings to file
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Log embedding stats
    stats = model.get_stats()
    
    return embeddings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate benchmark embeddings for SmartRouter")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (optional, will use environment variable if not provided)")
    parser.add_argument("--output", type=str, default="serverRouter/smartRouter/benchmark_embeddings.pkl", 
                       help="Path to save embeddings pickle file")
    parser.add_argument("--cache_dir", type=str, default="cache", 
                       help="Directory to cache embeddings")
    parser.add_argument("--env_file", type=str, default=".env",
                       help="Path to .env file containing API keys")
    return parser.parse_args()

def load_env(env_file=".env"):
    """Load environment variables from .env file if present."""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip("'").strip('"')
        return True
    return False

if __name__ == "__main__":
    args = parse_args()
    
    # Try to load from .env file first
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_env = os.path.join(repo_root, args.env_file)
    if os.path.exists(root_env):
        load_env(root_env)
    else:
        load_env(args.env_file)
    
    # Use API key from args or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("No OpenAI API key found. Please ensure it's in your .env file as OPENAI_API_KEY=your_key or use --api_key.")
        sys.exit(1)
    
    generate_benchmark_embeddings(
        output_file=args.output,
        api_key=api_key,
        cache_dir=args.cache_dir
    )