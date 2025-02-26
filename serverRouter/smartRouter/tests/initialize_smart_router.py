"""
Smart Router Initialization Script

This script initializes the Smart Router environment by:
1. Creating required directories
2. Setting up the benchmark embeddings
3. Testing basic functionality

Run this script from the serverRouter directory:
    python initialize_smart_router.py
"""

import os
import sys
import logging
from pathlib import Path
import pickle
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env(env_file=".env"):
    """Load environment variables from .env file."""
    if os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip("'").strip('"')
            return True
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
    else:
        logger.warning(f"No .env file found at {env_file}")
    return False

def check_openai_key():
    """Check if OpenAI API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found in environment variables")
        print("\nPlease set your OpenAI API key in the .env file or as an environment variable.")
        print("Example .env file content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    return True

def create_directories():
    """Create necessary directories for the smart router."""
    # Ensure smartRouter directory exists
    smart_router_dir = Path("smartRouter")
    smart_router_dir.mkdir(exist_ok=True)
    
    # Create cache directory for embeddings
    cache_dir = smart_router_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created directory structure at {smart_router_dir.absolute()}")
    return cache_dir

def generate_embeddings(cache_dir):
    """Generate benchmark embeddings."""
    try:
        # Add current directory to path for imports
        sys.path.append(str(Path.cwd()))
        
        # Import the embedding generator
        from smartRouter.embedding_model import OpenAIEmbeddings
        
        # Import benchmark data
        from smartRouter.benchmark_embedding_generator import BENCHMARKS, generate_benchmark_embeddings
        
        # Output file for embeddings
        output_file = "smartRouter/benchmark_embeddings.pkl"
        
        # Generate the embeddings
        logger.info("Generating benchmark embeddings...")
        embeddings = generate_benchmark_embeddings(
            output_file=output_file,
            cache_dir=str(cache_dir)
        )
        
        logger.info(f"Successfully generated embeddings for {len(embeddings)} benchmarks")
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_models():
    """Verify that models can be loaded."""
    try:
        # Try different import approaches
        try:
            # Try direct import first
            from core.models import CHAT_MODELS
        except ImportError:
            try:
                # Try relative import
                from .core.models import CHAT_MODELS
            except (ImportError, ValueError):
                # Direct import with corrected path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.append(current_dir)
                from core.models import CHAT_MODELS
        
        logger.info(f"Successfully loaded {len(CHAT_MODELS)} chat models")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_router():
    """Verify that the router can be initialized."""
    try:
        # Import the router
        from smartRouter.smart_router import SmartRouter
        
        # Initialize the router
        router = SmartRouter()
        
        logger.info("Successfully initialized the smart router")
        return True
    except Exception as e:
        logger.error(f"Error initializing router: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("SMART ROUTER INITIALIZATION".center(80))
    print("="*80 + "\n")
    
    # Step 1: Load environment variables
    root_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    local_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(root_env):
        load_env(root_env)
    elif os.path.exists(local_env):
        load_env(local_env)
    else:
        load_env()  # Try default location
    
    # Step 2: Check OpenAI API key
    if not check_openai_key():
        return
    
    # Step 3: Create directories
    cache_dir = create_directories()
    
    # Step 4: Verify models can be loaded
    if not verify_models():
        print("\nError loading models. Please check your models.py file.")
        return
    
    # Step 5: Generate embeddings
    print("\nGenerating benchmark embeddings (this may take a moment)...")
    if not generate_embeddings(cache_dir):
        print("\nError generating embeddings. Please check your OpenAI API key and try again.")
        return
    
    # Step 6: Verify router initialization
    if not verify_router():
        print("\nError initializing router. Please check the error messages above.")
        return
    
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE!".center(80))
    print("="*80)
    print("\nYou can now use the Smart Router in your application.")
    print("Run a quick test with: python standalone_test.py 'Your query here'")

if __name__ == "__main__":
    main()