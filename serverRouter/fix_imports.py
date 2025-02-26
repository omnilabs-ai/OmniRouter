"""
Fix imports for serverRouter project

This script patches import issues by creating symbolic links in the Python path
to make serverRouter modules available.
"""

import os
import sys
import inspect
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_imports():
    """Fix imports by adding symbolic links to sys.modules."""
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the parent directory (should be OmniRouter/)
    parent_dir = current_dir.parent
    
    # Create a symbolic module for serverRouter
    if 'serverRouter' not in sys.modules:
        import types
        sys.modules['serverRouter'] = types.ModuleType('serverRouter')
        sys.modules['serverRouter'].__path__ = [str(current_dir)]
        logger.info("Created serverRouter module")
    
    # Create symbolic modules for serverRouter.core
    if 'serverRouter.core' not in sys.modules:
        import types
        sys.modules['serverRouter.core'] = types.ModuleType('serverRouter.core')
        sys.modules['serverRouter.core'].__path__ = [str(current_dir / 'core')]
        logger.info("Created serverRouter.core module")
    
    # Create symbolic modules for serverRouter.smartRouter
    if 'serverRouter.smartRouter' not in sys.modules:
        import types
        sys.modules['serverRouter.smartRouter'] = types.ModuleType('serverRouter.smartRouter')
        sys.modules['serverRouter.smartRouter'].__path__ = [str(current_dir / 'smartRouter')]
        logger.info("Created serverRouter.smartRouter module")
    
    # Add the parent directory to sys.path
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
        logger.info(f"Added {parent_dir} to Python path")
    
    # Add the current directory to sys.path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.info(f"Added {current_dir} to Python path")
    
    return True

if __name__ == "__main__":
    fix_imports()
    print("Import fixes applied. You can now import serverRouter modules.")