from decimal import Decimal

import pytest

from omni.cost import UnknownPriceError, cost_of


def test_cost_with_cached_tokens():
    result = cost_of(
        {"input": 1_000_000, "output": 100_000, "cached_input": 200_000},
        {"pricing": {"input": 1, "output": 5, "cached_input": 0.25}},
    )
    assert result == Decimal("1.35")


def test_cost_without_cached_tokens():
    result = cost_of(
        {"input": 1_000_000, "output": 100_000},
        {"pricing": {"input": 1, "output": 5}},
    )
    assert result == Decimal("1.5")


def test_cached_tokens_cannot_exceed_input():
    with pytest.raises(UnknownPriceError):
        cost_of(
            {"input": 100, "cached_input": 200},
            {"pricing": {"input": 1, "cached_input": 0.25}},
        )


def test_missing_price_is_rejected():
    with pytest.raises(UnknownPriceError):
        cost_of(
            {"input": 100},
            {"pricing": {}},
        )
