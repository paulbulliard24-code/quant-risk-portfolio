"""Value-at-Risk and Expected Shortfall under three estimation methods.

Sign convention used everywhere in this project: losses are POSITIVE
numbers. A daily loss is minus the daily return, and VaR and ES are
reported as positive loss amounts (in return units, e.g. 0.02 = 2% of
portfolio value).

Definitions at confidence level c (e.g. 0.99):
- VaR is the loss that is only exceeded on (1-c) of days.
- ES is the average loss on the days when VaR is exceeded, so it
  describes how bad the tail is, not just where it starts.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class RiskEstimate:
    """VaR and ES at one confidence level, as positive loss fractions."""

    value_at_risk: float
    expected_shortfall: float


def historical_var_es(
    portfolio_returns: np.ndarray, confidence_level: float
) -> RiskEstimate:
    """Estimate VaR and ES by historical simulation.

    No distribution is assumed: VaR is read straight off the empirical
    quantile of past losses, and ES is the average of the losses beyond
    that quantile. The method captures fat tails that actually occurred,
    but can say nothing about losses worse than anything in the sample.

    Args:
        portfolio_returns: daily portfolio returns (gains positive).
        confidence_level: e.g. 0.95 or 0.99.

    Returns:
        RiskEstimate with VaR and ES as positive loss fractions.
    """
    losses = -np.asarray(portfolio_returns)
    value_at_risk = float(np.percentile(losses, 100.0 * confidence_level))
    tail_losses = losses[losses >= value_at_risk]
    expected_shortfall = float(np.mean(tail_losses))
    return RiskEstimate(value_at_risk, expected_shortfall)


def parametric_var_es(
    portfolio_returns: np.ndarray, confidence_level: float
) -> RiskEstimate:
    """Estimate VaR and ES assuming normally distributed returns.

    The variance-covariance method: fit a normal distribution to the
    return history and read the quantile analytically.
        VaR = -mean + std * z_c
        ES  = -mean + std * pdf(z_c) / (1 - c)
    The ES formula is the closed-form mean of a normal tail. Normality
    understates equity tails, so this method typically underestimates
    99% VaR (the backtest quantifies by how much).

    Args:
        portfolio_returns: daily portfolio returns (gains positive).
        confidence_level: e.g. 0.95 or 0.99.

    Returns:
        RiskEstimate with VaR and ES as positive loss fractions.
    """
    mean_return = float(np.mean(portfolio_returns))
    return_std = float(np.std(portfolio_returns, ddof=1))
    z_score = stats.norm.ppf(confidence_level)

    value_at_risk = -mean_return + return_std * z_score
    # Mean of the normal tail beyond the z-score quantile.
    tail_probability = 1.0 - confidence_level
    expected_shortfall = (
        -mean_return + return_std * stats.norm.pdf(z_score) / tail_probability
    )
    return RiskEstimate(value_at_risk, expected_shortfall)


def monte_carlo_var_es(
    asset_returns: np.ndarray,
    portfolio_weights: np.ndarray,
    confidence_level: float,
    n_simulations: int,
    seed: int,
) -> RiskEstimate:
    """Estimate VaR and ES by Monte Carlo from the asset covariance matrix.

    One-day asset returns are drawn from a multivariate normal fitted to
    history, preserving the correlations between assets, then aggregated
    into portfolio returns. With a normal driver this converges to the
    parametric answer as n_simulations grows; its value is that the
    driver can be swapped for a fat-tailed or scenario-based one without
    touching the rest of the pipeline.

    Args:
        asset_returns: history, shape (n_days, n_assets), gains positive.
        portfolio_weights: weight per asset, summing to 1.
        confidence_level: e.g. 0.95 or 0.99.
        n_simulations: number of simulated one-day scenarios.
        seed: random seed for reproducibility.

    Returns:
        RiskEstimate with VaR and ES as positive loss fractions.
    """
    random_generator = np.random.default_rng(seed)
    mean_returns = np.mean(asset_returns, axis=0)
    covariance_matrix = np.cov(asset_returns, rowvar=False)

    simulated_asset_returns = random_generator.multivariate_normal(
        mean_returns, covariance_matrix, size=n_simulations
    )
    # Portfolio return is the weighted average of asset returns.
    simulated_portfolio_returns = simulated_asset_returns @ portfolio_weights
    return historical_var_es(simulated_portfolio_returns, confidence_level)


def scale_var_to_horizon(one_day_var: float, horizon_days: int) -> float:
    """Scale a 1-day VaR to a longer horizon by the square root of time.

        VaR_h = VaR_1day * sqrt(h)

    This is the standard regulatory shortcut, and it rests on a shaky
    assumption: it is only exact if daily returns are independent and
    identically distributed. Real returns show volatility clustering
    (calm and turbulent regimes persist, as a GARCH model captures), so
    after a loss, further large losses are MORE likely than iid implies
    and sqrt-of-time understates multi-day risk exactly when it matters.
    Mean reversion has the opposite bias. Use with caution.

    Args:
        one_day_var: 1-day VaR as a positive loss fraction.
        horizon_days: target horizon in trading days.

    Returns:
        Scaled VaR as a positive loss fraction.
    """
    return one_day_var * np.sqrt(horizon_days)
