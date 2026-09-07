"""Cox-Ross-Rubinstein binomial tree for European and American options.

The tree discretises the risk-neutral dynamics into up/down moves and
prices by backward induction. Its distinctive value over the closed
form is AMERICAN exercise: at every node the holder chooses the better
of continuing or exercising immediately, a comparison a closed-form
European price cannot express.
"""

import numpy as np


def intrinsic_value(
    asset_prices: np.ndarray, strike: float, option_type: str
) -> np.ndarray:
    """Value of exercising immediately at the given asset prices.

    Args:
        asset_prices: asset price at each tree node.
        strike: option strike.
        option_type: "call" or "put".

    Returns:
        Exercise value per node (never negative).
    """
    if option_type == "call":
        return np.maximum(asset_prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - asset_prices, 0.0)
    raise ValueError(f"Unknown option type: {option_type}")


def binomial_tree_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
    exercise_style: str,
    n_steps: int,
) -> float:
    """Price an option on a Cox-Ross-Rubinstein binomial tree.

    CRR parameterisation: up = exp(sigma*sqrt(dt)), down = 1/up, and the
    risk-neutral up-probability p makes the expected asset growth equal
    the risk-free rate net of dividends — no view on direction enters,
    only volatility. Backward induction discounts the expected value one
    step at a time; for American style, each node's value is floored at
    immediate exercise.

    Args:
        spot: current asset price.
        strike: option strike.
        time_to_expiry: time to expiry in years.
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        volatility: annualised volatility.
        option_type: "call" or "put".
        exercise_style: "european" or "american".
        n_steps: number of tree steps (more steps, finer price grid).

    Returns:
        Option price in currency units.
    """
    if exercise_style not in ("european", "american"):
        raise ValueError(f"Unknown exercise style: {exercise_style}")

    step_years = time_to_expiry / n_steps
    up_factor = np.exp(volatility * np.sqrt(step_years))
    down_factor = 1.0 / up_factor
    # Risk-neutral probability: makes the expected one-step growth equal
    # the risk-free rate net of the dividend yield.
    growth_per_step = np.exp((risk_free_rate - dividend_yield) * step_years)
    up_probability = (growth_per_step - down_factor) / (up_factor - down_factor)
    discount_per_step = np.exp(-risk_free_rate * step_years)

    # Node j at a given step has had j up-moves: price = S * u^j * d^(steps-j).
    n_up_moves = np.arange(n_steps + 1)
    terminal_prices = spot * up_factor**n_up_moves * down_factor ** (
        n_steps - n_up_moves
    )
    option_values = intrinsic_value(terminal_prices, strike, option_type)

    for step in range(n_steps - 1, -1, -1):
        # Expected discounted value: node j's children are j (down) and
        # j+1 (up) at the next step.
        option_values = discount_per_step * (
            up_probability * option_values[1:]
            + (1.0 - up_probability) * option_values[:-1]
        )
        if exercise_style == "american":
            n_up_moves = np.arange(step + 1)
            node_prices = spot * up_factor**n_up_moves * down_factor ** (
                step - n_up_moves
            )
            # The holder exercises early whenever immediate exercise
            # beats the value of keeping the option alive.
            option_values = np.maximum(
                option_values, intrinsic_value(node_prices, strike, option_type)
            )
    return float(option_values[0])
