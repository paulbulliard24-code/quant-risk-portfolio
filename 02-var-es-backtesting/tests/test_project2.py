"""Tests for the VaR/ES engine and its backtesting machinery.

Each test checks a property that must hold by construction, so a
failure points at a real implementation bug rather than at noise.
"""

import numpy as np
from scipy import stats

from src.backtesting import kupiec_pof_test
from src.var_models import (
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
)

RANDOM_SEED = 42


def test_parametric_var_matches_analytical_value() -> None:
    """On genuinely normal data the parametric method must recover the
    textbook answer VaR = -mu + sigma * z within sampling error."""
    true_mean = 0.0005
    true_std = 0.02
    random_generator = np.random.default_rng(RANDOM_SEED)
    returns = random_generator.normal(true_mean, true_std, size=1_000_000)

    estimate = parametric_var_es(returns, confidence_level=0.99)
    analytical_var = -true_mean + true_std * stats.norm.ppf(0.99)
    assert np.isclose(estimate.value_at_risk, analytical_var, rtol=0.01)

    analytical_es = -true_mean + true_std * stats.norm.pdf(
        stats.norm.ppf(0.99)
    ) / 0.01
    assert np.isclose(estimate.expected_shortfall, analytical_es, rtol=0.01)


def test_expected_shortfall_is_at_least_var() -> None:
    """ES is the average loss BEYOND VaR, so ES >= VaR must hold for
    every method at every confidence level."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    asset_returns = random_generator.normal(0.0, 0.02, size=(1_000, 4))
    portfolio_weights = np.full(4, 0.25)
    portfolio_returns = asset_returns @ portfolio_weights

    for confidence_level in (0.95, 0.99):
        estimates = [
            historical_var_es(portfolio_returns, confidence_level),
            parametric_var_es(portfolio_returns, confidence_level),
            monte_carlo_var_es(
                asset_returns, portfolio_weights, confidence_level, 10_000, 7
            ),
        ]
        for estimate in estimates:
            assert estimate.expected_shortfall >= estimate.value_at_risk


def test_historical_var_extreme_confidence_boundary() -> None:
    """At 99.9% confidence on only 200 observations the empirical
    quantile sits at the edge of the sample: VaR cannot exceed the worst
    observed loss, and ES (the average beyond VaR) still tops it."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    returns = random_generator.normal(0.0, 0.02, size=200)
    worst_observed_loss = -np.min(returns)

    estimate = historical_var_es(returns, confidence_level=0.999)
    assert estimate.value_at_risk <= worst_observed_loss
    assert estimate.value_at_risk <= estimate.expected_shortfall
    assert estimate.expected_shortfall <= worst_observed_loss + 1e-12
    # And it must sit above the quantile at a lower confidence level.
    lower_confidence_var = historical_var_es(returns, 0.95).value_at_risk
    assert estimate.value_at_risk >= lower_confidence_var


def test_kupiec_rejects_miscalibrated_model() -> None:
    """A model claiming 99% coverage whose VaR is far too low produces
    many more than 1% exceptions; Kupiec must reject it decisively,
    while a correctly sized exception count must NOT be rejected."""
    n_days = 500
    random_generator = np.random.default_rng(RANDOM_SEED)
    losses = -random_generator.normal(0.0, 0.02, size=n_days)

    # VaR set at roughly the 69th percentile of losses instead of the
    # 99th: about 30% of days become exceptions.
    far_too_low_var = 0.01
    n_exceptions = int(np.sum(losses > far_too_low_var))
    assert n_exceptions > 0.1 * n_days  # confirm gross miscalibration

    rejection = kupiec_pof_test(n_exceptions, n_days, confidence_level=0.99)
    assert rejection.p_value < 0.001

    # Sanity check: an exception count exactly at the promised 1% rate
    # is fully consistent with the null.
    acceptance = kupiec_pof_test(5, n_days, confidence_level=0.99)
    assert acceptance.p_value > 0.05


def test_monte_carlo_with_fixed_seed_is_reproducible() -> None:
    """The same seed must give identical VaR and ES; a different seed
    must give a (slightly) different Monte Carlo answer."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    asset_returns = random_generator.normal(0.0, 0.02, size=(500, 3))
    portfolio_weights = np.array([0.5, 0.3, 0.2])

    first_run = monte_carlo_var_es(
        asset_returns, portfolio_weights, 0.99, 5_000, seed=11
    )
    second_run = monte_carlo_var_es(
        asset_returns, portfolio_weights, 0.99, 5_000, seed=11
    )
    different_seed_run = monte_carlo_var_es(
        asset_returns, portfolio_weights, 0.99, 5_000, seed=12
    )
    assert first_run.value_at_risk == second_run.value_at_risk
    assert first_run.expected_shortfall == second_run.expected_shortfall
    assert first_run.value_at_risk != different_seed_run.value_at_risk
