"""Replay: provider adapters and the single-turn replay runner (design doc section 2.4)."""

from omni.replay.adapters import (
    AnthropicAdapter,
    FixtureAdapter,
    MissingApiKey,
    OpenAICompatAdapter,
    ProviderAdapter,
    ReplayRequest,
    ReplayResponse,
    TransientError,
    UnsupportedParameter,
    adapter_for,
)
from omni.replay.runner import Candidate, RunSummary, run_replay

__all__ = [
    "AnthropicAdapter",
    "Candidate",
    "FixtureAdapter",
    "MissingApiKey",
    "OpenAICompatAdapter",
    "ProviderAdapter",
    "ReplayRequest",
    "ReplayResponse",
    "RunSummary",
    "TransientError",
    "UnsupportedParameter",
    "adapter_for",
    "run_replay",
]
