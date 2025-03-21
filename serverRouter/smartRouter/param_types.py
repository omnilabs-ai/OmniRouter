from enum import Enum
from typing import Union, Dict, Any

class LatencyType(str, Enum):
    """
    Defines latency preference types that users can select.
    Each type corresponds to a maximum latency value for the model.

    Latency is measured in seconds to first token.
    """
    LIGHTNING = "lightning"
    FAST = "fast"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @property
    def value_in_seconds(self) -> float:
        mapping = {
            LatencyType.LIGHTNING: 0.5,
            LatencyType.FAST: 0.8,
            LatencyType.BALANCED: 1.5,
            LatencyType.PERFORMANCE: 5.0
        }
        return mapping[self]

    @classmethod
    def from_value(cls, value: Union[str, float, "LatencyType"]) -> Union[float, "LatencyType"]:
        """Convert a string, float, or enum to the appropriate value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                valid_options = [e.name for e in cls]
                raise ValueError(f"Invalid latency type: {value}. Choose from {valid_options}")
        return value  # Return as is if it's a numeric value


class CostType(str, Enum):
    """
    Defines cost preference types that users can select.
    Each type corresponds to a maximum cost value for the model.

    Cost is measured in dollars per million tokens.
    """
    CHEAP = "cheap"
    BALANCED = "balanced"
    PREMIUM = "premium"
    PERFORMANCE = "performance"

    @property
    def value_in_dollars(self) -> float:
        mapping = {
            CostType.CHEAP: 1.5,
            CostType.BALANCED: 10.0,
            CostType.PREMIUM: 30.0,
            CostType.PERFORMANCE: 100.0
        }
        return mapping[self]

    @classmethod
    def from_value(cls, value: Union[str, float, "CostType"]) -> Union[float, "CostType"]:
        """Convert a string, float, or enum to the appropriate value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                valid_options = [e.name for e in cls]
                raise ValueError(f"Invalid cost type: {value}. Choose from {valid_options}")
        return value  # Return as is if it's a numeric value


