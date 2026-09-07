"""Backtesting a VaR model: does it deliver the coverage it promises?

A 99% VaR model is not judged by its numbers looking plausible; it is
judged by counting. Out of sample, actual losses should exceed the VaR
forecast on close to 1% of days (the "exceptions"), and those exception
days should be scattered, not clustered. This module implements:

- a rolling out-of-sample backtest,
- the Kupiec proportion-of-failures test (right NUMBER of exceptions),
- the Christoffersen independence test (no CLUSTERING of exceptions),
- the Basel traffic light classification used by regulators.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.var_models import (
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
)


@dataclass
class BacktestResult:
    """Out-of-sample VaR forecasts and what actually happened."""

    test_dates: pd.DatetimeIndex
    var_forecasts: np.ndarray  # positive loss fractions
    actual_losses: np.ndarray  # positive loss fractions
    exception_flags: np.ndarray  # True where actual loss exceeded VaR
    confidence_level: float


@dataclass
class TestResult:
    """A likelihood ratio statistic and its chi-squared p-value."""

    statistic: float
    p_value: float


def estimate_var_for_window(
    window_asset_returns: np.ndarray,
    portfolio_weights: np.ndarray,
    method_name: str,
    confidence_level: float,
    n_simulations: int,
    seed: int,
) -> float:
    """Estimate 1-day VaR from one estimation window with one method.

    Args:
        window_asset_returns: shape (window_days, n_assets).
        portfolio_weights: weight per asset, summing to 1.
        method_name: "historical", "parametric" or "monte_carlo".
        confidence_level: e.g. 0.95 or 0.99.
        n_simulations: Monte Carlo scenario count (ignored otherwise).
        seed: Monte Carlo seed (ignored otherwise).

    Returns:
        VaR as a positive loss fraction.
    """
    portfolio_returns = window_asset_returns @ portfolio_weights
    if method_name == "historical":
        return historical_var_es(portfolio_returns, confidence_level).value_at_risk
    if method_name == "parametric":
        return parametric_var_es(portfolio_returns, confidence_level).value_at_risk
    if method_name == "monte_carlo":
        return monte_carlo_var_es(
            window_asset_returns,
            portfolio_weights,
            confidence_level,
            n_simulations,
            seed,
        ).value_at_risk
    raise ValueError(f"Unknown VaR method: {method_name}")


def rolling_backtest(
    asset_returns: pd.DataFrame,
    portfolio_weights: np.ndarray,
    method_name: str,
    confidence_level: float,
    window_days: int = 250,
    n_simulations: int = 5_000,
    seed: int = 42,
) -> BacktestResult:
    """Run an out-of-sample rolling window backtest of one VaR method.

    For each test day, VaR is estimated on the preceding window_days of
    data only — the model never sees the day it is tested on — then the
    window rolls forward one day. An exception is a day whose realised
    loss exceeded the forecast.

    Args:
        asset_returns: daily asset returns indexed by date.
        portfolio_weights: weight per asset, summing to 1.
        method_name: "historical", "parametric" or "monte_carlo".
        confidence_level: e.g. 0.95 or 0.99.
        window_days: estimation window length (250 = one trading year).
        n_simulations: Monte Carlo scenario count per window.
        seed: base Monte Carlo seed; offset per day for fresh draws.

    Returns:
        BacktestResult over all days after the first window.
    """
    return_values = asset_returns.to_numpy()
    n_days = len(asset_returns)

    var_forecasts = np.zeros(n_days - window_days)
    actual_losses = np.zeros(n_days - window_days)
    for test_day in range(window_days, n_days):
        estimation_window = return_values[test_day - window_days : test_day]
        var_forecasts[test_day - window_days] = estimate_var_for_window(
            estimation_window,
            portfolio_weights,
            method_name,
            confidence_level,
            n_simulations,
            # Different seed per day, but reproducible across runs.
            seed + test_day,
        )
        realised_portfolio_return = return_values[test_day] @ portfolio_weights
        actual_losses[test_day - window_days] = -realised_portfolio_return

    return BacktestResult(
        test_dates=asset_returns.index[window_days:],
        var_forecasts=var_forecasts,
        actual_losses=actual_losses,
        exception_flags=actual_losses > var_forecasts,
        confidence_level=confidence_level,
    )


def count_times_log(count: float, probability: float) -> float:
    """Compute count * ln(probability), treating 0 * ln(0) as 0.

    In likelihood ratio tests a category with zero observations
    contributes nothing to the log-likelihood, but naive evaluation
    would produce 0 * (-inf) = NaN; this helper handles that limit.

    Args:
        count: number of observations in the category.
        probability: the category's probability under the model.

    Returns:
        The log-likelihood contribution.
    """
    if count == 0:
        return 0.0
    return count * np.log(probability)


def kupiec_pof_test(
    n_exceptions: int, n_observations: int, confidence_level: float
) -> TestResult:
    """Kupiec proportion-of-failures test: is the exception COUNT right?

    Null hypothesis: the true exception probability equals the promised
    (1 - confidence_level). The likelihood ratio compares the promised
    probability against the observed exception frequency; under the null
    it is chi-squared with 1 degree of freedom. A small p-value means
    the model's coverage is wrong (too many OR too few exceptions —
    too few wastes capital, too many understates risk).

    Args:
        n_exceptions: observed number of exception days.
        n_observations: number of backtest days.
        confidence_level: the VaR confidence level being tested.

    Returns:
        TestResult with the LR statistic and chi-squared(1) p-value.
    """
    promised_rate = 1.0 - confidence_level
    observed_rate = n_exceptions / n_observations
    n_non_exceptions = n_observations - n_exceptions

    log_likelihood_null = count_times_log(
        n_non_exceptions, 1.0 - promised_rate
    ) + count_times_log(n_exceptions, promised_rate)
    log_likelihood_observed = count_times_log(
        n_non_exceptions, 1.0 - observed_rate
    ) + count_times_log(n_exceptions, observed_rate)

    statistic = -2.0 * (log_likelihood_null - log_likelihood_observed)
    p_value = float(stats.chi2.sf(statistic, df=1))
    return TestResult(statistic, p_value)


def christoffersen_independence_test(
    exception_flags: np.ndarray,
) -> TestResult:
    """Christoffersen test: are exceptions independent or clustered?

    A model can have the right exception count but fail in bursts —
    several exceptions in one turbulent week — which is far more
    dangerous than isolated bad days. The test compares the probability
    of an exception following an exception against the probability of an
    exception following a quiet day; under independence they are equal,
    and the likelihood ratio is chi-squared with 1 degree of freedom.

    Args:
        exception_flags: boolean series of exception days, in time order.

    Returns:
        TestResult with the LR statistic and chi-squared(1) p-value.
    """
    flags = np.asarray(exception_flags, dtype=int)
    previous_day, current_day = flags[:-1], flags[1:]
    # Transition counts: n_ab = days moving from state a to state b.
    n_00 = int(np.sum((previous_day == 0) & (current_day == 0)))
    n_01 = int(np.sum((previous_day == 0) & (current_day == 1)))
    n_10 = int(np.sum((previous_day == 1) & (current_day == 0)))
    n_11 = int(np.sum((previous_day == 1) & (current_day == 1)))

    # Exception probability after a quiet day, after an exception day,
    # and unconditionally. Guard against empty categories.
    prob_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    prob_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0
    prob_any = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    log_likelihood_independent = count_times_log(
        n_00 + n_10, 1.0 - prob_any
    ) + count_times_log(n_01 + n_11, prob_any)
    log_likelihood_markov = (
        count_times_log(n_00, 1.0 - prob_01)
        + count_times_log(n_01, prob_01)
        + count_times_log(n_10, 1.0 - prob_11)
        + count_times_log(n_11, prob_11)
    )

    statistic = -2.0 * (log_likelihood_independent - log_likelihood_markov)
    p_value = float(stats.chi2.sf(statistic, df=1))
    return TestResult(statistic, p_value)


def basel_traffic_light_zone(exception_flags: np.ndarray) -> str:
    """Classify a 99% VaR model into the Basel traffic light zones.

    Regulators count exceptions over the most recent 250 trading days of
    99% VaR backtesting: 0-4 is green (model accepted), 5-9 is amber
    (capital multiplier increases), 10 or more is red (model presumed
    flawed). The thresholds are calibrated so a correct model lands in
    green with high probability (expected count is 2.5).

    Args:
        exception_flags: boolean exception series at 99% confidence; the
            most recent 250 entries are used.

    Returns:
        "green", "amber" or "red".
    """
    recent_exceptions = int(np.sum(exception_flags[-250:]))
    if recent_exceptions <= 4:
        return "green"
    if recent_exceptions <= 9:
        return "amber"
    return "red"
