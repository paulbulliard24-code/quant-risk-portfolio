"""Tests for the option pricing implementations and their validation.

Each test pins the code to an external anchor (a textbook value, a
model-free identity, a convergence law), so passing means the pricers
are right, not merely self-consistent.
"""

import numpy as np
import pytest

from src.binomial_tree import binomial_tree_price
from src.black_scholes import option_price
from src.implied_vol import implied_volatility
from src.monte_carlo import monte_carlo_price
from src.validation import (
    compare_greeks,
    finite_difference_first_order_greeks,
    finite_difference_second_order_greeks,
    put_call_parity_error,
)

# Base scenario used unless a test needs something specific.
SPOT = 100.0
STRIKE = 100.0
TIME_TO_EXPIRY = 0.5
RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.02
VOLATILITY = 0.25


def test_black_scholes_matches_hull_textbook_example() -> None:
    """Hull, "Options, Futures, and Other Derivatives" (9th edition),
    Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5, no dividends.
    Hull reports call = 4.76 and put = 0.81."""
    call = option_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, "call")
    put = option_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, "put")
    assert np.isclose(call, 4.76, atol=0.005)
    assert np.isclose(put, 0.81, atol=0.005)


def test_put_call_parity_holds_to_machine_precision() -> None:
    """C - P = S*e^(-qT) - K*e^(-rT) is model-free, so the closed-form
    prices must satisfy it to floating point accuracy, at every strike
    and maturity."""
    for strike in (70.0, 100.0, 130.0):
        for maturity in (0.1, 1.0, 3.0):
            parity_error = put_call_parity_error(
                SPOT, strike, maturity, RISK_FREE_RATE, DIVIDEND_YIELD,
                VOLATILITY,
            )
            assert abs(parity_error) < 1e-10


def test_monte_carlo_price_within_confidence_interval() -> None:
    """The Monte Carlo estimate must land within its own reported 95%
    confidence interval of the closed form — the engine's error bars
    have to be honest, not just its point estimate."""
    for option_type in ("call", "put"):
        closed_form = option_price(
            SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
            VOLATILITY, option_type,
        )
        result = monte_carlo_price(
            SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
            VOLATILITY, option_type, n_simulations=1_000_000, seed=42,
        )
        assert result.ci_lower <= closed_form <= result.ci_upper


def test_binomial_tree_converges_to_black_scholes() -> None:
    """Tree error is a discretisation bias of order 1/n_steps, so a
    fine tree must beat a coarse one and land close to the closed form."""
    closed_form = option_price(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call",
    )
    coarse_price = binomial_tree_price(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call", "european", n_steps=10,
    )
    fine_price = binomial_tree_price(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call", "european", n_steps=2_000,
    )
    assert abs(fine_price - closed_form) < abs(coarse_price - closed_form)
    assert abs(fine_price - closed_form) < 0.005


def test_analytical_greeks_match_finite_differences() -> None:
    """The analytical formulas and central finite differences are two
    independent computations of the same derivatives; in market units
    they must agree to 1e-4 for every Greek, calls and puts alike."""
    for option_type in ("call", "put"):
        comparison = compare_greeks(
            SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
            VOLATILITY, option_type,
        )
        for _, row in comparison.iterrows():
            assert row["abs_difference"] < 1e-4, (
                f"{option_type} {row['greek']}: analytical "
                f"{row['analytical']:.8f} vs finite difference "
                f"{row['finite_difference']:.8f}"
            )


def test_implied_volatility_round_trip() -> None:
    """Pricing at a known volatility and inverting the price must
    recover that volatility almost exactly; and a quote outside the
    no-arbitrage bounds must raise a clear error, not fail numerically."""
    true_volatility = 0.23
    for option_type in ("call", "put"):
        price = option_price(
            SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
            true_volatility, option_type,
        )
        recovered = implied_volatility(
            price, SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE,
            DIVIDEND_YIELD, option_type,
        )
        assert abs(recovered - true_volatility) < 1e-8

    # A call can never be worth more than the asset: no volatility can
    # reproduce such a quote and the solver must say so.
    with pytest.raises(ValueError):
        implied_volatility(
            SPOT + 10.0, SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE,
            DIVIDEND_YIELD, "call",
        )


def test_gamma_and_vega_identical_for_call_and_put() -> None:
    """Put-call parity says C - P is linear in spot and independent of
    volatility, so its second spot-derivative (gamma) and its
    volatility-derivative (vega) are zero: call and put must share both.
    Checked via finite differences on the two prices separately, so the
    test does not simply call one shared formula twice."""
    call_second_order = finite_difference_second_order_greeks(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call",
    )
    put_second_order = finite_difference_second_order_greeks(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "put",
    )
    assert abs(call_second_order["gamma"] - put_second_order["gamma"]) < 1e-8

    call_first_order = finite_difference_first_order_greeks(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call",
    )
    put_first_order = finite_difference_first_order_greeks(
        SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "put",
    )
    assert abs(call_first_order["vega"] - put_first_order["vega"]) < 1e-8
