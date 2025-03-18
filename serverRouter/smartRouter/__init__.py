"""
Smart Router Package

This package provides the SmartRouter system for intelligent model selection.
"""

from . import config
from . import smart_router
from . import task_vector_db
from . import session_tracker
from . import embedding_model

from serverRouter.smartRouter.smart_router import SmartRouter
from serverRouter.smartRouter.config import SmartRouterConfig

__all__ = ["SmartRouter", "SmartRouterConfig"]
