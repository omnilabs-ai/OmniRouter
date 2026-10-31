"""Model usage cost calculator. Prices are USD per 1 million tokens."""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation


class UnknownPriceError(ValueError):
    """Raised when pricing information is missing or invalid."""


def cost_of(
    usage: Mapping[str, int | float],
    model: Mapping[str, object],
) -> Decimal:
    """Calculate request cost in USD using per-million-token prices.

    Supported usage keys: input, output, cached_input, reasoning.
    Optional pricing key: per_request (USD per request).
    Cached input tokens are assumed to be included in input tokens.
    """

    pricing = model.get("pricing")
    if not isinstance(pricing, Mapping):
        raise UnknownPriceError("Model pricing is missing.")

    def number(value: object, label: str) -> Decimal:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise UnknownPriceError(f"Invalid {label}: {value!r}") from None
        if not result.is_finite() or result < 0:
            raise UnknownPriceError(f"{label} must be a finite non-negative number.")
        return result

    def tokens(key: str) -> Decimal:
        value = number(usage.get(key, 0), f"usage.{key}")
        if value != value.to_integral_value():
            raise UnknownPriceError(f"usage.{key} must be a whole number.")
        return value

    input_tokens = tokens("input")
    output_tokens = tokens("output")
    cached_tokens = tokens("cached_input")
    reasoning_tokens = tokens("reasoning")

    if cached_tokens > input_tokens:
        raise UnknownPriceError("cached_input cannot exceed input.")

    total = number(pricing.get("per_request", 0), "pricing.per_request")

    # Cached tokens are charged at cached_input rate; remaining input at input rate.
    uncached_tokens = input_tokens - cached_tokens

    for token_count, price_key in (
        (uncached_tokens, "input"),
        (cached_tokens, "cached_input"),
        (output_tokens, "output"),
        (reasoning_tokens, "reasoning"),
    ):
        if token_count == 0:
            continue
        if price_key not in pricing:
            raise UnknownPriceError(f"Missing pricing.{price_key}.")
        price = number(pricing[price_key], f"pricing.{price_key}")
        total += token_count * price / Decimal("1000000")

    return total
