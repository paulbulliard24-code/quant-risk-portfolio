"""Cross-checks between the three pricing implementations.

Model validation logic: no single implementation is trusted. The closed
form, the Monte Carlo engine and the binomial tree are three independent
routes to the same number, so genuine agreement across a grid of
strikes and maturities is strong evidence that each one is right — a
shared bug in three unrelated algorithms is unlikely. The same idea is
applied to Greeks (analytical formula vs finite differences) and to
model-free identities (put-call parity, American exercise bounds).
"""

import numpy as np
import pandas as pd

from src.binomial_tree import binomial_tree_price
from src.black_scholes import (
    delta,
    gamma,
    option_price,
    rho_per_basis_point,
    theta_per_day,
    vanna_per_vol_point,
    vega_per_vol_point,
    volga_per_vol_point,
)
from src.monte_carlo import monte_carlo_price

# Finite difference bump sizes. Bumps must be small enough that the
# truncation error (proportional to bump^2) is negligible, but large
# enough that the price difference is not swamped by floating point
# round-off.
SPOT_BUMP = 0.5
VOLATILITY_BUMP = 0.01
RATE_BUMP = 0.0001
ONE_DAY = 1.0 / 365.0


def compare_methods_on_grid(
    strikes: list[float],
    maturities: list[float],
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
    n_mc_simulations: int,
    n_tree_steps: int,
    tree_tolerance: float,
    seed: int,
) -> pd.DataFrame:
    """Compare Monte Carlo and binomial prices to the closed form.

    Pass criteria: the Monte Carlo estimate must lie within 3 standard
    errors of the closed form (99.7% band, so a correct engine passes a
    28-cell grid without spurious failures); the binomial price must lie
    within a fixed absolute tolerance.

    Args:
        strikes: strike grid.
        maturities: maturity grid in years.
        spot, risk_free_rate, dividend_yield, volatility: market inputs.
        option_type: "call" or "put".
        n_mc_simulations: Monte Carlo scenarios per grid point.
        n_tree_steps: binomial steps per grid point.
        tree_tolerance: max acceptable |binomial - closed form|.
        seed: base Monte Carlo seed, offset per grid point.

    Returns:
        One row per (strike, maturity, method) with price, difference
        from the closed form, tolerance and pass/fail.
    """
    grid_points = []
    for strike in strikes:
        for maturity in maturities:
            grid_points.append((strike, maturity))

    rows = []
    for grid_index, (strike, maturity) in enumerate(grid_points):
        closed_form = option_price(
            spot, strike, maturity, risk_free_rate, dividend_yield,
            volatility, option_type,
        )
        mc_result = monte_carlo_price(
            spot, strike, maturity, risk_free_rate, dividend_yield,
            volatility, option_type, n_mc_simulations, seed + grid_index,
        )
        tree_price = binomial_tree_price(
            spot, strike, maturity, risk_free_rate, dividend_yield,
            volatility, option_type, "european", n_tree_steps,
        )
        rows.append({
            "strike": strike, "maturity": maturity, "method": "monte_carlo",
            "closed_form": closed_form, "price": mc_result.price,
            "abs_difference": abs(mc_result.price - closed_form),
            "tolerance": 3.0 * mc_result.standard_error,
            "passed": abs(mc_result.price - closed_form)
            <= 3.0 * mc_result.standard_error,
        })
        rows.append({
            "strike": strike, "maturity": maturity, "method": "binomial",
            "closed_form": closed_form, "price": tree_price,
            "abs_difference": abs(tree_price - closed_form),
            "tolerance": tree_tolerance,
            "passed": abs(tree_price - closed_form) <= tree_tolerance,
        })
    return pd.DataFrame(rows)


def monte_carlo_convergence(
    n_simulations_list: list[int],
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
    seed: int,
) -> pd.DataFrame:
    """Measure Monte Carlo error as the simulation count grows.

    Monte Carlo error is statistical, so it should fall as 1/sqrt(n):
    four times the scenarios buys only half the error. Both the actual
    error against the closed form and the engine's own reported
    standard error are recorded, so the plot can confirm they shrink
    together at the theoretical rate.

    Args: single option definition plus the list of scenario counts.

    Returns:
        One row per simulation count with the price, absolute error and
        reported standard error.
    """
    closed_form = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, option_type,
    )
    rows = []
    for n_simulations in n_simulations_list:
        result = monte_carlo_price(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
            volatility, option_type, n_simulations, seed,
        )
        rows.append({
            "n_simulations": n_simulations,
            "price": result.price,
            "abs_error": abs(result.price - closed_form),
            "standard_error": result.standard_error,
        })
    return pd.DataFrame(rows)


def binomial_convergence(
    n_steps_list: list[int],
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> pd.DataFrame:
    """Measure binomial tree error as the number of steps grows.

    Tree error is a discretisation bias, not statistical noise: it
    shrinks roughly as 1/n_steps, oscillating as the strike moves
    between grid nodes — a signature worth recognising when validating
    lattice models.

    Args: single option definition plus the list of step counts.

    Returns:
        One row per step count with the price and absolute error.
    """
    closed_form = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, option_type,
    )
    rows = []
    for n_steps in n_steps_list:
        tree_price = binomial_tree_price(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
            volatility, option_type, "european", n_steps,
        )
        rows.append({
            "n_steps": n_steps,
            "price": tree_price,
            "abs_error": abs(tree_price - closed_form),
        })
    return pd.DataFrame(rows)


def finite_difference_first_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> dict[str, float]:
    """First-order Greeks by central differences, in market units.

    Central differences (bump up and down, divide by twice the bump)
    cancel the leading error term, leaving an error proportional to the
    bump squared. Theta uses a SHORTER expiry for the up-bump because
    theta is the derivative with respect to the passage of time, which
    reduces time-to-expiry.

    Args: as in black_scholes.option_price.

    Returns:
        {"delta", "vega", "theta", "rho"} in the same units as the
        analytical functions (per vol point, per day, per basis point).
    """
    def price_at(bumped_spot, bumped_expiry, bumped_rate, bumped_vol):
        """Reprice with one input bumped, holding the others fixed."""
        return option_price(
            bumped_spot, strike, bumped_expiry, bumped_rate,
            dividend_yield, bumped_vol, option_type,
        )

    delta_fd = (
        price_at(spot + SPOT_BUMP, time_to_expiry, risk_free_rate, volatility)
        - price_at(spot - SPOT_BUMP, time_to_expiry, risk_free_rate, volatility)
    ) / (2.0 * SPOT_BUMP)
    # Bumping volatility by one point gives "per vol point" directly.
    vega_fd = (
        price_at(spot, time_to_expiry, risk_free_rate, volatility + VOLATILITY_BUMP)
        - price_at(spot, time_to_expiry, risk_free_rate, volatility - VOLATILITY_BUMP)
    ) / 2.0
    theta_fd = (
        price_at(spot, time_to_expiry - ONE_DAY, risk_free_rate, volatility)
        - price_at(spot, time_to_expiry + ONE_DAY, risk_free_rate, volatility)
    ) / 2.0
    # Bumping the rate by one basis point gives "per basis point" directly.
    rho_fd = (
        price_at(spot, time_to_expiry, risk_free_rate + RATE_BUMP, volatility)
        - price_at(spot, time_to_expiry, risk_free_rate - RATE_BUMP, volatility)
    ) / 2.0
    return {"delta": delta_fd, "vega": vega_fd, "theta": theta_fd, "rho": rho_fd}


def finite_difference_second_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> dict[str, float]:
    """Second-order Greeks by finite differences, in market units.

    Gamma and volga are second differences in one variable; vanna is a
    cross difference (bump spot and volatility in all four combinations).

    Args: as in black_scholes.option_price.

    Returns:
        {"gamma", "vanna", "volga"} in the analytical functions' units.
    """
    def price_at(bumped_spot, bumped_vol):
        """Reprice with spot and/or volatility bumped."""
        return option_price(
            bumped_spot, strike, time_to_expiry, risk_free_rate,
            dividend_yield, bumped_vol, option_type,
        )

    base_price = price_at(spot, volatility)
    gamma_fd = (
        price_at(spot + SPOT_BUMP, volatility)
        - 2.0 * base_price
        + price_at(spot - SPOT_BUMP, volatility)
    ) / SPOT_BUMP**2
    # Cross difference divided by (2*spot bump)*(2*vol bump); the vol
    # bump of one point makes the result "per vol point" directly.
    vanna_fd = (
        price_at(spot + SPOT_BUMP, volatility + VOLATILITY_BUMP)
        - price_at(spot + SPOT_BUMP, volatility - VOLATILITY_BUMP)
        - price_at(spot - SPOT_BUMP, volatility + VOLATILITY_BUMP)
        + price_at(spot - SPOT_BUMP, volatility - VOLATILITY_BUMP)
    ) / (4.0 * SPOT_BUMP)
    volga_fd = (
        price_at(spot, volatility + VOLATILITY_BUMP)
        - 2.0 * base_price
        + price_at(spot, volatility - VOLATILITY_BUMP)
    )
    return {"gamma": gamma_fd, "vanna": vanna_fd, "volga": volga_fd}


def compare_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> pd.DataFrame:
    """Tabulate analytical vs finite-difference Greeks and their gap.

    Args: as in black_scholes.option_price.

    Returns:
        One row per Greek: analytical value, finite-difference value,
        absolute difference. All in market-convention units.
    """
    market_inputs = (
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility,
    )
    analytical = {
        "delta": delta(*market_inputs, option_type),
        "gamma": gamma(*market_inputs),
        "vega": vega_per_vol_point(*market_inputs),
        "theta": theta_per_day(*market_inputs, option_type),
        "rho": rho_per_basis_point(*market_inputs, option_type),
        "vanna": vanna_per_vol_point(*market_inputs),
        "volga": volga_per_vol_point(*market_inputs),
    }
    finite_difference = finite_difference_first_order_greeks(
        *market_inputs, option_type
    ) | finite_difference_second_order_greeks(*market_inputs, option_type)

    rows = []
    for greek_name in analytical:
        rows.append({
            "greek": greek_name,
            "analytical": analytical[greek_name],
            "finite_difference": finite_difference[greek_name],
            "abs_difference": abs(
                analytical[greek_name] - finite_difference[greek_name]
            ),
        })
    return pd.DataFrame(rows)


def put_call_parity_error(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Deviation from put-call parity: C - P - (S e^(-qT) - K e^(-rT)).

    Parity is model-free — it follows from no-arbitrage alone, without
    any distributional assumption — so a violation is an unambiguous
    implementation bug, never a modelling choice.

    Args: as in black_scholes.option_price (no option_type).

    Returns:
        The parity residual (should be zero to machine precision).
    """
    call = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, "call",
    )
    put = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, "put",
    )
    forward_leg = spot * np.exp(-dividend_yield * time_to_expiry) - (
        strike * np.exp(-risk_free_rate * time_to_expiry)
    )
    return call - put - forward_leg


def american_exercise_checks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_tree_steps: int,
) -> dict[str, float]:
    """Check the two classic American exercise relationships.

    1. An American put is worth MORE than a European put when rates are
       positive: exercising early converts the option into strike cash
       that earns interest, and deep in the money that beats waiting.
    2. An American call on a NON-dividend asset equals the European
       call: early exercise forfeits remaining optionality and pays the
       strike early, both losses, so the right is worthless.

    Args: as in black_scholes.option_price, plus tree step count. The
        dividend_yield applies to the put check; the call check forces
        dividend_yield to zero, which the relationship requires.

    Returns:
        Prices for both checks and the two differences.
    """
    european_put = binomial_tree_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, "put", "european", n_tree_steps,
    )
    american_put = binomial_tree_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        volatility, "put", "american", n_tree_steps,
    )
    european_call_no_dividend = binomial_tree_price(
        spot, strike, time_to_expiry, risk_free_rate, 0.0,
        volatility, "call", "european", n_tree_steps,
    )
    american_call_no_dividend = binomial_tree_price(
        spot, strike, time_to_expiry, risk_free_rate, 0.0,
        volatility, "call", "american", n_tree_steps,
    )
    return {
        "european_put": european_put,
        "american_put": american_put,
        "early_exercise_premium": american_put - european_put,
        "european_call_no_dividend": european_call_no_dividend,
        "american_call_no_dividend": american_call_no_dividend,
        "call_difference": abs(
            american_call_no_dividend - european_call_no_dividend
        ),
    }
