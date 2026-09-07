"""Monte Carlo pricing of European options by risk-neutral simulation.

For a European payoff only the terminal price matters, so the whole
path need not be simulated: the terminal price under the risk-neutral
measure is drawn in one step,

    S_T = S_0 * exp((r - q - sigma^2/2) * T + sigma * sqrt(T) * Z).

The -sigma^2/2 term is the Ito correction: it converts the drift of the
price into the drift of the log price, so that the simulated asset
grows at exactly the risk-free rate (net of dividends) on average.
Omitting it is the classic Monte Carlo pricing bug — prices come out
biased high by roughly half the variance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    """A simulated price with its statistical uncertainty."""

    price: float
    standard_error: float
    ci_lower: float  # 95% confidence interval bounds
    ci_upper: float


def option_payoff(
    terminal_prices: np.ndarray, strike: float, option_type: str
) -> np.ndarray:
    """Payoff of a European option at expiry.

    Args:
        terminal_prices: simulated asset prices at expiry.
        strike: option strike.
        option_type: "call" or "put".

    Returns:
        Payoff per scenario (never negative — options are rights).
    """
    if option_type == "call":
        return np.maximum(terminal_prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - terminal_prices, 0.0)
    raise ValueError(f"Unknown option type: {option_type}")


def simulate_terminal_prices(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    standard_normals: np.ndarray,
) -> np.ndarray:
    """Map standard normal draws to risk-neutral terminal prices.

    Args:
        spot: current asset price.
        time_to_expiry: time to expiry in years.
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        volatility: annualised volatility.
        standard_normals: Z draws, one per scenario.

    Returns:
        Terminal asset prices, one per scenario.
    """
    # Risk-neutral log drift with the Ito correction -sigma^2/2.
    log_drift = (
        risk_free_rate - dividend_yield - 0.5 * volatility**2
    ) * time_to_expiry
    log_diffusion = volatility * np.sqrt(time_to_expiry) * standard_normals
    return spot * np.exp(log_drift + log_diffusion)


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
    n_simulations: int,
    seed: int,
) -> MonteCarloResult:
    """Price a European option by antithetic Monte Carlo.

    Antithetic variates: each draw Z is paired with -Z. The two payoffs
    are negatively correlated (a high price in one scenario is a low
    price in its mirror), so the average of each pair has lower variance
    than two independent draws — a free variance reduction. The standard
    error is computed over the PAIR averages, which is the statistically
    correct unit of independence.

    Args:
        spot: current asset price.
        strike: option strike.
        time_to_expiry: time to expiry in years.
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        volatility: annualised volatility.
        option_type: "call" or "put".
        n_simulations: total scenario count (rounded down to even).
        seed: random seed for reproducibility.

    Returns:
        MonteCarloResult with the price and its 95% confidence interval.
    """
    random_generator = np.random.default_rng(seed)
    n_pairs = n_simulations // 2
    standard_normals = random_generator.standard_normal(n_pairs)

    prices_up = simulate_terminal_prices(
        spot, time_to_expiry, risk_free_rate, dividend_yield, volatility,
        standard_normals,
    )
    prices_down = simulate_terminal_prices(
        spot, time_to_expiry, risk_free_rate, dividend_yield, volatility,
        -standard_normals,
    )
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    pair_average_payoffs = discount_factor * 0.5 * (
        option_payoff(prices_up, strike, option_type)
        + option_payoff(prices_down, strike, option_type)
    )

    price = float(np.mean(pair_average_payoffs))
    standard_error = float(
        np.std(pair_average_payoffs, ddof=1) / np.sqrt(n_pairs)
    )
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        ci_lower=price - 1.96 * standard_error,
        ci_upper=price + 1.96 * standard_error,
    )
