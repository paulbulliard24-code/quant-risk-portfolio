"""Futures pricing under the Schwartz (1997) one-factor model.

The hedge in this project is a short position in a futures contract with
a FIXED maturity date. As the hedge horizon passes, the contract's
time-to-maturity shrinks day by day, so the futures price must be
recomputed each day from the simulated spot price and the remaining
time-to-maturity.
"""

import numpy as np

from src.price_simulation import compute_log_price_mean


def schwartz_futures_price(
    spot_price: np.ndarray | float,
    time_to_maturity_years: float,
    long_run_log_price: float,
    mean_reversion_speed: float,
    volatility: float,
) -> np.ndarray | float:
    """Closed-form futures price under the Schwartz one-factor model.

        F(S, T) = exp( exp(-kappa*T) * ln(S)
                       + (1 - exp(-kappa*T)) * alpha
                       + (sigma^2 / (4*kappa)) * (1 - exp(-2*kappa*T)) )

    The futures price is a weighted blend of today's log spot price and
    the long-run mean alpha: the further away maturity is, the more the
    market expects mean reversion to have pulled the price back, so the
    less weight today's spot gets. This is why long-dated futures move
    less than spot ("Samuelson effect") under mean reversion.

    Args:
        spot_price: current spot price (scalar or array of paths).
        time_to_maturity_years: T, time until the futures contract matures.
        long_run_log_price: mu, the long-run level parameter.
        mean_reversion_speed: kappa, speed of mean reversion (per year).
        volatility: sigma, annualised volatility of the log price.

    Returns:
        Futures price(s), same shape as spot_price.
    """
    log_price_mean = compute_log_price_mean(
        long_run_log_price, mean_reversion_speed, volatility
    )

    # Weight on today's spot: 1 at maturity, decaying towards 0 for
    # long-dated contracts as mean reversion dominates.
    spot_weight = np.exp(-mean_reversion_speed * time_to_maturity_years)

    # Half the conditional variance of ln(S_T): a Jensen adjustment,
    # because the futures price is the expectation of the price, not the
    # exponential of the expected log price.
    variance_adjustment = (volatility**2 / (4.0 * mean_reversion_speed)) * (
        1.0 - np.exp(-2.0 * mean_reversion_speed * time_to_maturity_years)
    )

    log_futures_price = (
        spot_weight * np.log(spot_price)
        + (1.0 - spot_weight) * log_price_mean
        + variance_adjustment
    )
    return np.exp(log_futures_price)


def schwartz_futures_paths(
    spot_paths: np.ndarray,
    initial_time_to_maturity_years: float,
    time_step_years: float,
    long_run_log_price: float,
    mean_reversion_speed: float,
    volatility: float,
) -> np.ndarray:
    """Mark a fixed-maturity futures contract to market along spot paths.

    The contract matures on a fixed calendar date, so on day d the
    remaining time-to-maturity is the initial maturity minus d steps
    (floored at zero once the contract has matured).

    Args:
        spot_paths: simulated spot prices, shape (n_paths, n_days + 1).
        initial_time_to_maturity_years: time-to-maturity on day 0.
        time_step_years: length of one step in years (1/252 for daily).
        long_run_log_price: mu, the long-run level parameter.
        mean_reversion_speed: kappa, speed of mean reversion (per year).
        volatility: sigma, annualised volatility of the log price.

    Returns:
        Futures prices with the same shape as spot_paths.
    """
    n_days_plus_one = spot_paths.shape[1]
    futures_paths = np.zeros_like(spot_paths)
    for day in range(n_days_plus_one):
        time_to_maturity = max(
            initial_time_to_maturity_years - day * time_step_years, 0.0
        )
        futures_paths[:, day] = schwartz_futures_price(
            spot_paths[:, day],
            time_to_maturity,
            long_run_log_price,
            mean_reversion_speed,
            volatility,
        )
    return futures_paths
