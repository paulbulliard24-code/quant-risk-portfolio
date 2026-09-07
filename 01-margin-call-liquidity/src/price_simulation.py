"""Spot price simulation for a commodity.

Two models are implemented:

1. Schwartz (1997) one-factor model: the log price mean-reverts. This
   matches energy commodities, where high prices attract new supply and
   destroy demand, pulling the price back towards a long-run level.
2. Geometric Brownian motion (GBM): the standard non-mean-reverting
   baseline, included to show how much mean reversion matters for
   liquidity risk.

Both simulators return an array with one row per path and one column per
day (column 0 is today's spot price), and both take a seed so every
result in this project is reproducible.
"""

import numpy as np


def compute_log_price_mean(
    long_run_log_price: float,
    mean_reversion_speed: float,
    volatility: float,
) -> float:
    """Compute alpha, the level that the log price mean-reverts towards.

    In the Schwartz model the log price X = ln(S) reverts to
    alpha = mu - sigma^2 / (2 * kappa). The subtraction is a convexity
    (Jensen) adjustment: because the price is the exponential of X,
    volatility in X pushes the average price level up, and alpha is set
    below mu to compensate.

    Args:
        long_run_log_price: mu, the model's long-run level parameter.
        mean_reversion_speed: kappa, the speed of mean reversion (per year).
        volatility: sigma, annualised volatility of the log price.

    Returns:
        alpha, the long-run mean of the log price.
    """
    convexity_adjustment = volatility**2 / (2.0 * mean_reversion_speed)
    return long_run_log_price - convexity_adjustment


def simulate_schwartz_paths(
    spot_price: float,
    long_run_log_price: float,
    mean_reversion_speed: float,
    volatility: float,
    n_days: int,
    n_paths: int,
    time_step_years: float,
    seed: int,
) -> np.ndarray:
    """Simulate spot price paths under the Schwartz one-factor model.

    The log price X = ln(S) follows an Ornstein-Uhlenbeck process:
        dX = kappa * (alpha - X) dt + sigma dW.
    The exact discretisation is used (not Euler), so the simulation is
    correct for any step size:
        X_next = X * exp(-kappa*dt) + alpha * (1 - exp(-kappa*dt))
                 + sigma * sqrt((1 - exp(-2*kappa*dt)) / (2*kappa)) * Z.

    Args:
        spot_price: today's spot price S_0.
        long_run_log_price: mu, the long-run level parameter.
        mean_reversion_speed: kappa, speed of mean reversion (per year).
        volatility: sigma, annualised volatility of the log price.
        n_days: number of daily steps to simulate.
        n_paths: number of independent paths.
        time_step_years: length of one step in years (1/252 for daily).
        seed: random seed for reproducibility.

    Returns:
        Array of prices with shape (n_paths, n_days + 1); column 0 is spot.
    """
    random_generator = np.random.default_rng(seed)
    log_price_mean = compute_log_price_mean(
        long_run_log_price, mean_reversion_speed, volatility
    )

    # How much of yesterday's deviation from the long-run mean survives
    # one day of mean reversion.
    decay_factor = np.exp(-mean_reversion_speed * time_step_years)

    # Exact one-step standard deviation of the OU process: shocks are
    # partly pulled back within the day, so this is below sigma*sqrt(dt).
    step_std_dev = volatility * np.sqrt(
        (1.0 - np.exp(-2.0 * mean_reversion_speed * time_step_years))
        / (2.0 * mean_reversion_speed)
    )

    log_prices = np.zeros((n_paths, n_days + 1))
    log_prices[:, 0] = np.log(spot_price)
    for day in range(1, n_days + 1):
        random_shocks = random_generator.standard_normal(n_paths)
        log_prices[:, day] = (
            log_prices[:, day - 1] * decay_factor
            + log_price_mean * (1.0 - decay_factor)
            + step_std_dev * random_shocks
        )
    return np.exp(log_prices)


def simulate_gbm_paths(
    spot_price: float,
    drift: float,
    volatility: float,
    n_days: int,
    n_paths: int,
    time_step_years: float,
    seed: int,
) -> np.ndarray:
    """Simulate spot price paths under geometric Brownian motion.

    GBM has no mean reversion: a price shock is permanent, so shocks
    accumulate and the price can drift arbitrarily far from its start.
    This makes it a natural worst-case comparison for margin risk.

    Args:
        spot_price: today's spot price S_0.
        drift: annualised expected return mu of the price.
        volatility: sigma, annualised volatility of returns.
        n_days: number of daily steps to simulate.
        n_paths: number of independent paths.
        time_step_years: length of one step in years (1/252 for daily).
        seed: random seed for reproducibility.

    Returns:
        Array of prices with shape (n_paths, n_days + 1); column 0 is spot.
    """
    random_generator = np.random.default_rng(seed)

    # Exact GBM log-price step: the -sigma^2/2 term converts the price
    # drift into the drift of the log price.
    log_drift_per_step = (drift - 0.5 * volatility**2) * time_step_years
    step_std_dev = volatility * np.sqrt(time_step_years)

    log_prices = np.zeros((n_paths, n_days + 1))
    log_prices[:, 0] = np.log(spot_price)
    for day in range(1, n_days + 1):
        random_shocks = random_generator.standard_normal(n_paths)
        log_prices[:, day] = (
            log_prices[:, day - 1]
            + log_drift_per_step
            + step_std_dev * random_shocks
        )
    return np.exp(log_prices)
