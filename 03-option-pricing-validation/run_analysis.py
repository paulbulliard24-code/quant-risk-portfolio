"""Independent validation of three option pricing implementations.

The closed-form Black-Scholes price, an antithetic Monte Carlo engine
and a Cox-Ross-Rubinstein binomial tree are three independent routes to
the same number. This script cross-checks them the way a model
validation team would: agreement across a strike/maturity grid,
convergence at the theoretically expected rates, analytical vs
finite-difference Greeks, model-free identities (put-call parity,
American exercise bounds) and an implied volatility round trip.

Outputs (written to outputs/):
- mc_convergence.png / binomial_convergence.png
- greeks_vs_spot.png
- vol_smile_roundtrip.png
- validation_summary.csv / greeks_comparison.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no display window needed
import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import (
    delta,
    gamma,
    option_price,
    theta_per_day,
    vega_per_vol_point,
)
from src.implied_vol import implied_volatility
from src.validation import (
    american_exercise_checks,
    binomial_convergence,
    compare_greeks,
    compare_methods_on_grid,
    monte_carlo_convergence,
    put_call_parity_error,
)

# Base market scenario. The dividend yield is deliberately non-zero so
# the q-dependent code paths are exercised everywhere.
SPOT = 100.0
RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.02
VOLATILITY = 0.25

STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
MATURITIES = [0.25, 0.5, 1.0, 2.0]
N_MC_SIMULATIONS = 200_000
N_TREE_STEPS = 1_000
TREE_TOLERANCE = 0.02
RANDOM_SEED = 42

OUTPUT_DIR = Path(__file__).parent / "outputs"


def plot_monte_carlo_convergence() -> None:
    """Plot Monte Carlo error vs scenario count against the 1/sqrt(n) law."""
    n_simulations_list = [1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000]
    convergence = monte_carlo_convergence(
        n_simulations_list, SPOT, 100.0, 1.0, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call", RANDOM_SEED,
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.loglog(
        convergence["n_simulations"], convergence["abs_error"],
        marker="o", label="actual |MC - closed form|",
    )
    axis.loglog(
        convergence["n_simulations"], convergence["standard_error"],
        marker="s", label="reported standard error",
    )
    # Reference: error halves each time n quadruples.
    reference = convergence["standard_error"].iloc[0] * np.sqrt(
        n_simulations_list[0] / convergence["n_simulations"]
    )
    axis.loglog(
        convergence["n_simulations"], reference,
        linestyle="--", color="grey", label="1/sqrt(n) reference",
    )
    axis.set_xlabel("Number of simulations")
    axis.set_ylabel("Error ($)")
    axis.set_title("Monte Carlo error falls as 1/sqrt(n)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mc_convergence.png", dpi=150)
    plt.close(figure)


def plot_binomial_convergence() -> None:
    """Plot the binomial price closing in on the closed form with steps."""
    n_steps_list = [5, 10, 25, 50, 100, 200, 400, 800, 1_600]
    convergence = binomial_convergence(
        n_steps_list, SPOT, 100.0, 1.0, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call",
    )
    closed_form = option_price(
        SPOT, 100.0, 1.0, RISK_FREE_RATE, DIVIDEND_YIELD, VOLATILITY, "call"
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        convergence["n_steps"], convergence["price"],
        marker="o", label="binomial price",
    )
    axis.axhline(
        closed_form, color="red", linestyle="--",
        label=f"closed form = {closed_form:.4f}",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Number of tree steps")
    axis.set_ylabel("Price ($)")
    axis.set_title("Binomial tree converges to the closed form (error ~ 1/n)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "binomial_convergence.png", dpi=150)
    plt.close(figure)


def plot_greeks_vs_spot() -> None:
    """Plot delta, gamma, vega and theta as functions of the spot price."""
    spot_grid = np.linspace(60.0, 140.0, 161)
    strike, maturity = 100.0, 0.5
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    greek_values = {"call delta": [], "put delta": [], "gamma": [],
                    "vega": [], "call theta": [], "put theta": []}
    for spot_value in spot_grid:
        inputs = (spot_value, strike, maturity, RISK_FREE_RATE,
                  DIVIDEND_YIELD, VOLATILITY)
        greek_values["call delta"].append(delta(*inputs, "call"))
        greek_values["put delta"].append(delta(*inputs, "put"))
        greek_values["gamma"].append(gamma(*inputs))
        greek_values["vega"].append(vega_per_vol_point(*inputs))
        greek_values["call theta"].append(theta_per_day(*inputs, "call"))
        greek_values["put theta"].append(theta_per_day(*inputs, "put"))

    axes[0, 0].plot(spot_grid, greek_values["call delta"], label="call")
    axes[0, 0].plot(spot_grid, greek_values["put delta"], label="put")
    axes[0, 0].set_title("Delta (per $1 of spot)")
    axes[0, 1].plot(spot_grid, greek_values["gamma"], color="purple")
    axes[0, 1].set_title("Gamma (per $1 of spot, call = put)")
    axes[1, 0].plot(spot_grid, greek_values["vega"], color="green")
    axes[1, 0].set_title("Vega (per vol point, call = put)")
    axes[1, 1].plot(spot_grid, greek_values["call theta"], label="call")
    axes[1, 1].plot(spot_grid, greek_values["put theta"], label="put")
    axes[1, 1].set_title("Theta (per calendar day)")
    for axis in axes.flat:
        axis.axvline(strike, color="grey", linestyle=":", linewidth=1)
        axis.set_xlabel("Spot price ($)")
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    figure.suptitle(f"Greeks vs spot (K={strike:.0f}, T={maturity}y)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "greeks_vs_spot.png", dpi=150)
    plt.close(figure)


def smile_volatility(strike: float) -> float:
    """A stylised implied volatility smile as a function of strike.

    Quadratic in log-moneyness: out-of-the-money strikes on both sides
    trade at higher implied volatility than at-the-money, the typical
    post-1987 equity index pattern.

    Args:
        strike: option strike.

    Returns:
        Implied volatility for that strike.
    """
    log_moneyness = np.log(strike / SPOT)
    return 0.18 + 0.4 * log_moneyness**2


def plot_smile_roundtrip() -> float:
    """Price options along a smile, invert back, and plot both curves.

    A round trip through the pricer and the implied volatility solver
    must reproduce the input smile exactly; any gap would reveal an
    inconsistency between the two.

    Returns:
        The worst absolute volatility error across the smile.
    """
    strikes = np.linspace(70.0, 130.0, 25)
    maturity = 0.5
    input_vols, recovered_vols = [], []
    for strike in strikes:
        input_vol = smile_volatility(strike)
        price = option_price(
            SPOT, strike, maturity, RISK_FREE_RATE, DIVIDEND_YIELD,
            input_vol, "call",
        )
        recovered_vol = implied_volatility(
            price, SPOT, strike, maturity, RISK_FREE_RATE, DIVIDEND_YIELD,
            "call",
        )
        input_vols.append(input_vol)
        recovered_vols.append(recovered_vol)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(strikes, 100 * np.array(input_vols), label="input smile")
    axis.plot(
        strikes, 100 * np.array(recovered_vols),
        marker="x", linestyle="none", label="recovered from prices",
    )
    axis.set_xlabel("Strike ($)")
    axis.set_ylabel("Implied volatility (%)")
    axis.set_title("Volatility smile round trip: price, invert, recover")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vol_smile_roundtrip.png", dpi=150)
    plt.close(figure)
    return float(np.max(np.abs(np.array(input_vols) - np.array(recovered_vols))))


def main() -> None:
    """Run every validation check and write all outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_monte_carlo_convergence()
    plot_binomial_convergence()
    plot_greeks_vs_spot()
    smile_error = plot_smile_roundtrip()

    grid_comparison = compare_methods_on_grid(
        STRIKES, MATURITIES, SPOT, RISK_FREE_RATE, DIVIDEND_YIELD,
        VOLATILITY, "call", N_MC_SIMULATIONS, N_TREE_STEPS, TREE_TOLERANCE,
        RANDOM_SEED,
    )
    grid_comparison.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)

    greeks_comparison = compare_greeks(
        SPOT, 100.0, 0.5, RISK_FREE_RATE, DIVIDEND_YIELD, VOLATILITY, "call"
    )
    greeks_comparison.to_csv(OUTPUT_DIR / "greeks_comparison.csv", index=False)

    parity_errors = []
    for strike in STRIKES:
        for maturity in MATURITIES:
            parity_errors.append(put_call_parity_error(
                SPOT, strike, maturity, RISK_FREE_RATE, DIVIDEND_YIELD,
                VOLATILITY,
            ))
    exercise_checks = american_exercise_checks(
        SPOT, 110.0, 1.0, RISK_FREE_RATE, DIVIDEND_YIELD, VOLATILITY,
        N_TREE_STEPS,
    )

    print("=" * 74)
    print("OPTION PRICING MODEL VALIDATION - SUMMARY")
    print("=" * 74)
    n_checks = len(grid_comparison)
    n_passed = int(grid_comparison["passed"].sum())
    print(f"Grid cross-check ({len(STRIKES)} strikes x {len(MATURITIES)} "
          f"maturities, call): {n_passed}/{n_checks} passed")
    print(f"  worst MC deviation:       "
          f"{grid_comparison[grid_comparison.method == 'monte_carlo'].abs_difference.max():.5f}")
    print(f"  worst binomial deviation: "
          f"{grid_comparison[grid_comparison.method == 'binomial'].abs_difference.max():.5f}"
          f" (tolerance {TREE_TOLERANCE})")
    print()
    print("Greeks: analytical vs finite difference (ATM call, T=0.5):")
    print(greeks_comparison.to_string(index=False, float_format="{:.8f}".format))
    print()
    print(f"Put-call parity: worst |error| across grid = "
          f"{np.max(np.abs(parity_errors)):.2e}")
    print(f"Smile round trip: worst |vol error| = {smile_error:.2e}")
    print()
    print("American exercise checks (1000-step tree):")
    print(f"  European put {exercise_checks['european_put']:.4f} vs American "
          f"put {exercise_checks['american_put']:.4f} -> early exercise "
          f"premium {exercise_checks['early_exercise_premium']:.4f} (must be > 0)")
    print(f"  European vs American call, q=0: difference "
          f"{exercise_checks['call_difference']:.2e} (must be ~ 0)")
    print()
    print(f"Figures and tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
