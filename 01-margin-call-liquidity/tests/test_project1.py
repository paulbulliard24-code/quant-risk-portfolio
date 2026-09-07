"""Tests for the margin call liquidity risk simulator.

Each test checks a property that must hold by construction of the model,
so a failure points at a real implementation bug rather than at noise.
"""

import numpy as np

from src.futures_pricing import schwartz_futures_paths, schwartz_futures_price
from src.margin_engine import daily_variation_margin
from src.price_simulation import (
    compute_log_price_mean,
    simulate_schwartz_paths,
)

# Shared illustrative parameters for the tests.
SPOT_PRICE = 80.0
LONG_RUN_LOG_PRICE = np.log(80.0)
MEAN_REVERSION_SPEED = 1.5
VOLATILITY = 0.35
TIME_STEP_YEARS = 1.0 / 252.0


def test_zero_volatility_produces_zero_margin_calls() -> None:
    """With no volatility and spot at its long-run level, prices never
    move, so the mark-to-market change and every margin call is zero."""
    spot_paths = simulate_schwartz_paths(
        spot_price=SPOT_PRICE,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=MEAN_REVERSION_SPEED,
        volatility=0.0,
        n_days=126,
        n_paths=5,
        time_step_years=TIME_STEP_YEARS,
        seed=1,
    )
    futures_paths = schwartz_futures_paths(
        spot_paths,
        initial_time_to_maturity_years=0.5,
        time_step_years=TIME_STEP_YEARS,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=MEAN_REVERSION_SPEED,
        volatility=0.0,
    )
    for path_index in range(futures_paths.shape[0]):
        margin_flows = daily_variation_margin(
            futures_paths[path_index], n_contracts=1_000, contract_size=1_000.0
        )
        assert np.allclose(margin_flows, 0.0)


def test_log_price_long_run_mean_converges_to_alpha() -> None:
    """After many years, mean reversion has erased the starting point, so
    the cross-path average log price must sit at alpha (the stationary
    mean), within Monte Carlo error."""
    n_years = 10
    n_days = n_years * 252
    spot_paths = simulate_schwartz_paths(
        spot_price=SPOT_PRICE * 2.0,  # start far from the long-run level
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=MEAN_REVERSION_SPEED,
        volatility=VOLATILITY,
        n_days=n_days,
        n_paths=20_000,
        time_step_years=TIME_STEP_YEARS,
        seed=7,
    )
    final_log_prices = np.log(spot_paths[:, -1])
    alpha = compute_log_price_mean(
        LONG_RUN_LOG_PRICE, MEAN_REVERSION_SPEED, VOLATILITY
    )
    # Stationary std of the log price is sigma / sqrt(2*kappa) ~ 0.20, so
    # the standard error of the mean over 20,000 paths is ~0.0014.
    assert abs(np.mean(final_log_prices) - alpha) < 0.01


def test_futures_price_converges_to_spot_at_maturity() -> None:
    """At maturity a futures contract settles against spot, so F(S, 0)
    must equal S exactly, and F for tiny maturities must be close."""
    for spot_price in (40.0, 80.0, 160.0):
        price_at_maturity = schwartz_futures_price(
            spot_price,
            time_to_maturity_years=0.0,
            long_run_log_price=LONG_RUN_LOG_PRICE,
            mean_reversion_speed=MEAN_REVERSION_SPEED,
            volatility=VOLATILITY,
        )
        assert np.isclose(price_at_maturity, spot_price)

        price_near_maturity = schwartz_futures_price(
            spot_price,
            time_to_maturity_years=1e-6,
            long_run_log_price=LONG_RUN_LOG_PRICE,
            mean_reversion_speed=MEAN_REVERSION_SPEED,
            volatility=VOLATILITY,
        )
        assert np.isclose(price_near_maturity, spot_price, rtol=1e-4)


def test_variation_margin_sums_to_total_position_value_change() -> None:
    """Daily settlement only changes the timing of cash flows: summed
    over the horizon, variation margin must equal the total change in
    value of the short position (the daily changes telescope)."""
    n_contracts = 1_000
    contract_size = 1_000.0
    spot_paths = simulate_schwartz_paths(
        spot_price=SPOT_PRICE,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=MEAN_REVERSION_SPEED,
        volatility=VOLATILITY,
        n_days=126,
        n_paths=50,
        time_step_years=TIME_STEP_YEARS,
        seed=3,
    )
    futures_paths = schwartz_futures_paths(
        spot_paths,
        initial_time_to_maturity_years=0.5,
        time_step_years=TIME_STEP_YEARS,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=MEAN_REVERSION_SPEED,
        volatility=VOLATILITY,
    )
    for path_index in range(futures_paths.shape[0]):
        futures_path = futures_paths[path_index]
        margin_flows = daily_variation_margin(
            futures_path, n_contracts, contract_size
        )
        # Short position value change over the whole horizon.
        total_value_change = (
            -n_contracts * contract_size * (futures_path[-1] - futures_path[0])
        )
        assert np.isclose(np.sum(margin_flows), total_value_change)


def test_same_seed_gives_identical_results() -> None:
    """Reproducibility: the same seed must give bit-identical paths, and
    a different seed must give different paths."""
    first_run = simulate_schwartz_paths(
        SPOT_PRICE,
        LONG_RUN_LOG_PRICE,
        MEAN_REVERSION_SPEED,
        VOLATILITY,
        126,
        100,
        TIME_STEP_YEARS,
        seed=42,
    )
    second_run = simulate_schwartz_paths(
        SPOT_PRICE,
        LONG_RUN_LOG_PRICE,
        MEAN_REVERSION_SPEED,
        VOLATILITY,
        126,
        100,
        TIME_STEP_YEARS,
        seed=42,
    )
    different_seed_run = simulate_schwartz_paths(
        SPOT_PRICE,
        LONG_RUN_LOG_PRICE,
        MEAN_REVERSION_SPEED,
        VOLATILITY,
        126,
        100,
        TIME_STEP_YEARS,
        seed=43,
    )
    assert np.array_equal(first_run, second_run)
    assert not np.array_equal(first_run, different_seed_run)
