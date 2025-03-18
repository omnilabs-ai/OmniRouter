#!/usr/bin/env python3
"""
Task Vector Database Generator

This script generates a vector database of task examples for SmartRouter task classification.
It creates embeddings for each example and saves the database to a file for future use.

Usage:
    python -m serverRouter.smartRouter.generate_task_db [--output PATH] [--export-json PATH]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Import the necessary modules
from serverRouter.smartRouter.embedding_model import OpenAIEmbeddings
from serverRouter.smartRouter.task_vector_db import TaskVectorDB, create_default_example_db

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate task vector database")
    parser.add_argument(
        "--output", 
        type=str, 
        default="serverRouter/smartRouter/task_examples_db.pkl",
        help="Path to save the database (default: serverRouter/smartRouter/task_examples_db.pkl)"
    )
    parser.add_argument(
        "--export-json", 
        type=str,
        help="Export examples to JSON file (optional)"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="serverRouter/smartRouter/cache",
        help="Directory to cache embeddings (default: serverRouter/smartRouter/cache)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-ada-002",
        help="OpenAI embedding model to use"
    )
    parser.add_argument(
        "--additional-examples", 
        type=str, 
        default=None,
        help="Path to JSON file with additional examples to import"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (optional, will use environment variable if not provided)"
    )
    return parser.parse_args()

def load_env(env_file: str = ".env") -> bool:
    """
    Load environment variables from .env file if present.
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        True if environment variables were loaded, False otherwise
    """
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip("'").strip('"')
            return True
        except Exception:
            return False
    return False

def main() -> None:
    """
    Generate and save the task vector database.
    
    This function:
    1. Initializes the embedding client
    2. Creates the database with default examples
    3. Imports additional examples if provided
    4. Exports examples to JSON for reference
    5. Saves the database to the specified path
    """
    args = parse_arguments()
    
    # Try to load environment variables from .env
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_env = os.path.join(repo_root, ".env")
    if os.path.exists(root_env):
        load_env(root_env)
    else:
        load_env()
    
    # Get API key from arguments or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found. Using dummy embeddings instead.")
    
    try:
        # Create embedding client
        print("Initializing embedding client...")
        embeddings_client = OpenAIEmbeddings(
            api_key=api_key,
            model=args.model,
            cache_dir=args.cache_dir
        )
        
        # Create the database with default examples
        print("Creating task vector database with default examples...")
        db = create_default_example_db(embeddings_client)
        
        # Import additional examples if provided
        if args.additional_examples and os.path.exists(args.additional_examples):
            print(f"Importing additional examples from {args.additional_examples}...")
            try:
                count = db.import_examples_json(args.additional_examples)
                print(f"Imported {count} additional examples")
            except Exception as e:
                print(f"Error importing additional examples: {e}")
        
        # Export examples to JSON for human-readable reference
        if args.export_json:
            print(f"Exporting examples to {args.export_json}...")
            try:
                # Create directory for export file if needed
                export_path = Path(args.export_json)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                
                db.export_examples_json(args.export_json)
            except Exception as e:
                print(f"Error exporting examples to JSON: {e}")
        
        # Save the database
        output_path = args.output
        print(f"Saving database to {output_path}...")
        try:
            # Create directory for output file if needed
            db_path = Path(output_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            db.save(output_path)
        except Exception as e:
            print(f"Error saving database: {e}")
            sys.exit(1)
        
        # Print statistics
        print(f"Task vector database created with {db.get_example_count()} examples")
        print(f"Task types: {', '.join(db.get_task_types())}")
        
        # Print example counts by task type
        for task_type in db.get_task_types():
            count = db.get_example_count(task_type)
            print(f"  - {task_type}: {count} examples")
        
        # Log embedding stats
        if hasattr(embeddings_client, 'get_stats'):
            stats = embeddings_client.get_stats()
            print(f"Embedding generation stats: {stats}")
            
        print("Done!")
    except Exception as e:
        print(f"Unexpected error generating task database: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 