"""Margin call liquidity risk analysis for a hedged commodity trader.

Scenario: a trader holds 1,000,000 barrels of physical oil and shorts
1,000 futures contracts against it. The hedge is value-neutral, but the
futures leg is marked to market daily. This script simulates how much
cash the margin calls can drain and sizes the committed credit line
needed to survive at 95% and 99% confidence (Liquidity-at-Risk).

Outputs (written to outputs/):
- sample_paths.png: simulated spot and futures paths
- peak_outflow_histogram.png: distribution of peak outflow with LaR lines
- model_comparison.csv: Schwartz vs GBM liquidity risk
- kappa_sensitivity.csv / kappa_sensitivity.png: LaR vs reversion speed
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.futures_pricing import schwartz_futures_paths
from src.liquidity_risk import liquidity_at_risk, simulate_peak_outflows
from src.price_simulation import simulate_gbm_paths, simulate_schwartz_paths

# --- Market and position parameters (illustrative, see README) ---------
SPOT_PRICE = 80.0  # $/barrel
LONG_RUN_LOG_PRICE = np.log(80.0)  # start at the long-run level
MEAN_REVERSION_SPEED = 1.5  # kappa: deviations halve in ~5.5 months
VOLATILITY = 0.35  # annualised log price volatility
GBM_DRIFT = 0.0  # zero drift so GBM differs only by lack of reversion

N_CONTRACTS = 1_000  # short futures contracts
CONTRACT_SIZE = 1_000.0  # barrels per contract (NYMEX WTI convention)
INITIAL_MARGIN_RATE = 0.10  # initial margin as fraction of notional

HEDGE_HORIZON_DAYS = 126  # six months of trading days
TIME_STEP_YEARS = 1.0 / 252.0
FUTURES_MATURITY_YEARS = 0.5  # contract matures at the end of the hedge

N_SIMULATIONS = 10_000
RANDOM_SEED = 42
CONFIDENCE_LEVELS = (0.95, 0.99)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def simulate_schwartz_futures_paths(
    mean_reversion_speed: float, seed: int
) -> np.ndarray:
    """Simulate Schwartz spot paths and mark the futures hedge on them.

    Args:
        mean_reversion_speed: kappa used for both spot and futures.
        seed: random seed for reproducibility.

    Returns:
        Futures prices, shape (N_SIMULATIONS, HEDGE_HORIZON_DAYS + 1).
    """
    spot_paths = simulate_schwartz_paths(
        spot_price=SPOT_PRICE,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=mean_reversion_speed,
        volatility=VOLATILITY,
        n_days=HEDGE_HORIZON_DAYS,
        n_paths=N_SIMULATIONS,
        time_step_years=TIME_STEP_YEARS,
        seed=seed,
    )
    return schwartz_futures_paths(
        spot_paths=spot_paths,
        initial_time_to_maturity_years=FUTURES_MATURITY_YEARS,
        time_step_years=TIME_STEP_YEARS,
        long_run_log_price=LONG_RUN_LOG_PRICE,
        mean_reversion_speed=mean_reversion_speed,
        volatility=VOLATILITY,
    )


def compute_liquidity_statistics(futures_paths: np.ndarray) -> dict[str, float]:
    """Reduce simulated futures paths to headline liquidity risk numbers.

    Args:
        futures_paths: futures prices, shape (n_paths, n_days + 1).

    Returns:
        Mean peak outflow and LaR at each confidence level, in dollars.
    """
    peak_outflows = simulate_peak_outflows(
        futures_paths, N_CONTRACTS, CONTRACT_SIZE, INITIAL_MARGIN_RATE
    )
    statistics = {"mean_peak_outflow": float(np.mean(peak_outflows))}
    for confidence_level in CONFIDENCE_LEVELS:
        statistics[f"lar_{int(confidence_level * 100)}"] = liquidity_at_risk(
            peak_outflows, confidence_level
        )
    return statistics


def plot_sample_paths(
    spot_paths: np.ndarray, futures_paths: np.ndarray, n_shown: int
) -> None:
    """Plot a handful of simulated spot and futures paths side by side.

    Args:
        spot_paths: simulated spot prices, shape (n_paths, n_days + 1).
        futures_paths: matching futures prices, same shape.
        n_shown: number of paths to draw on each panel.
    """
    figure, (spot_axis, futures_axis) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )
    days = np.arange(spot_paths.shape[1])
    for path_index in range(n_shown):
        spot_axis.plot(days, spot_paths[path_index], linewidth=0.8)
        futures_axis.plot(days, futures_paths[path_index], linewidth=0.8)
    spot_axis.set_title("Spot price (Schwartz one-factor)")
    futures_axis.set_title("Futures price (fixed maturity)")
    for axis in (spot_axis, futures_axis):
        axis.set_xlabel("Trading day")
        axis.axhline(SPOT_PRICE, color="black", linestyle="--", linewidth=1)
    spot_axis.set_ylabel("Price ($/barrel)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "sample_paths.png", dpi=150)
    plt.close(figure)


def plot_peak_outflow_histogram(
    peak_outflows: np.ndarray, lar_95: float, lar_99: float
) -> None:
    """Plot the peak outflow distribution with the LaR quantiles marked.

    Args:
        peak_outflows: peak margin outflow per simulated path, in dollars.
        lar_95: Liquidity-at-Risk at 95% confidence, in dollars.
        lar_99: Liquidity-at-Risk at 99% confidence, in dollars.
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(peak_outflows / 1e6, bins=80, color="steelblue", alpha=0.8)
    axis.axvline(
        lar_95 / 1e6, color="orange", linewidth=2, label="LaR 95%"
    )
    axis.axvline(lar_99 / 1e6, color="red", linewidth=2, label="LaR 99%")
    axis.set_xlabel("Peak cumulative margin outflow ($ millions)")
    axis.set_ylabel("Number of scenarios")
    axis.set_title(
        "Distribution of peak margin outflow over a 6-month hedge"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "peak_outflow_histogram.png", dpi=150)
    plt.close(figure)


def build_model_comparison_table() -> pd.DataFrame:
    """Compare liquidity risk under Schwartz vs GBM price dynamics.

    Under GBM the futures hedge is marked directly on the simulated
    price (no term structure), because without mean reversion and with
    zero rates the futures price simply tracks the spot price.

    Returns:
        One row per model with mean peak outflow and LaR statistics.
    """
    schwartz_futures = simulate_schwartz_futures_paths(
        MEAN_REVERSION_SPEED, RANDOM_SEED
    )
    gbm_futures = simulate_gbm_paths(
        spot_price=SPOT_PRICE,
        drift=GBM_DRIFT,
        volatility=VOLATILITY,
        n_days=HEDGE_HORIZON_DAYS,
        n_paths=N_SIMULATIONS,
        time_step_years=TIME_STEP_YEARS,
        seed=RANDOM_SEED,
    )
    rows = {
        "Schwartz (mean-reverting)": compute_liquidity_statistics(
            schwartz_futures
        ),
        "GBM (no mean reversion)": compute_liquidity_statistics(gbm_futures),
    }
    return pd.DataFrame(rows).T


def run_kappa_sensitivity(kappa_values: list[float]) -> pd.DataFrame:
    """Recompute LaR for a range of mean reversion speeds.

    Faster mean reversion caps how far prices can run away, and it
    shrinks the volatility of the futures price relative to spot, so
    LaR should fall as kappa rises.

    Args:
        kappa_values: mean reversion speeds to test (per year).

    Returns:
        One row per kappa with LaR at each confidence level.
    """
    rows = []
    for kappa in kappa_values:
        futures_paths = simulate_schwartz_futures_paths(kappa, RANDOM_SEED)
        statistics = compute_liquidity_statistics(futures_paths)
        statistics["kappa"] = kappa
        rows.append(statistics)
    return pd.DataFrame(rows).set_index("kappa")


def plot_kappa_sensitivity(sensitivity_table: pd.DataFrame) -> None:
    """Plot LaR as a function of the mean reversion speed kappa.

    Args:
        sensitivity_table: output of run_kappa_sensitivity.
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    for column, colour in (("lar_95", "orange"), ("lar_99", "red")):
        axis.plot(
            sensitivity_table.index,
            sensitivity_table[column] / 1e6,
            marker="o",
            color=colour,
            label=column.replace("lar_", "LaR ") + "%",
        )
    axis.set_xlabel("Mean reversion speed kappa (per year)")
    axis.set_ylabel("Liquidity-at-Risk ($ millions)")
    axis.set_title("Faster mean reversion lowers the credit line needed")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "kappa_sensitivity.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Run the full liquidity risk analysis and write all outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Sample paths for the figures (small run, separate seed is fine).
    spot_paths = simulate_schwartz_paths(
        SPOT_PRICE,
        LONG_RUN_LOG_PRICE,
        MEAN_REVERSION_SPEED,
        VOLATILITY,
        HEDGE_HORIZON_DAYS,
        N_SIMULATIONS,
        TIME_STEP_YEARS,
        RANDOM_SEED,
    )
    futures_paths = schwartz_futures_paths(
        spot_paths,
        FUTURES_MATURITY_YEARS,
        TIME_STEP_YEARS,
        LONG_RUN_LOG_PRICE,
        MEAN_REVERSION_SPEED,
        VOLATILITY,
    )
    plot_sample_paths(spot_paths, futures_paths, n_shown=20)

    peak_outflows = simulate_peak_outflows(
        futures_paths, N_CONTRACTS, CONTRACT_SIZE, INITIAL_MARGIN_RATE
    )
    lar_95 = liquidity_at_risk(peak_outflows, 0.95)
    lar_99 = liquidity_at_risk(peak_outflows, 0.99)
    plot_peak_outflow_histogram(peak_outflows, lar_95, lar_99)

    comparison_table = build_model_comparison_table()
    comparison_table.to_csv(OUTPUT_DIR / "model_comparison.csv")

    sensitivity_table = run_kappa_sensitivity([0.25, 0.5, 1.0, 1.5, 2.0, 4.0])
    sensitivity_table.to_csv(OUTPUT_DIR / "kappa_sensitivity.csv")
    plot_kappa_sensitivity(sensitivity_table)

    position_notional = SPOT_PRICE * N_CONTRACTS * CONTRACT_SIZE
    print("=" * 70)
    print("MARGIN CALL LIQUIDITY RISK - SUMMARY")
    print("=" * 70)
    print(f"Position: short {N_CONTRACTS:,} contracts "
          f"({N_CONTRACTS * CONTRACT_SIZE:,.0f} barrels), "
          f"notional ${position_notional / 1e6:,.0f}m")
    print(f"Horizon: {HEDGE_HORIZON_DAYS} trading days, "
          f"{N_SIMULATIONS:,} simulations, seed {RANDOM_SEED}")
    print(f"Mean peak outflow:  ${np.mean(peak_outflows) / 1e6:8.1f}m")
    print(f"LaR 95%:            ${lar_95 / 1e6:8.1f}m")
    print(f"LaR 99%:            ${lar_99 / 1e6:8.1f}m")
    print()
    print("Model comparison ($):")
    print(comparison_table.to_string(float_format="{:,.0f}".format))
    print()
    print("Kappa sensitivity ($):")
    print(sensitivity_table.to_string(float_format="{:,.0f}".format))
    print()
    print(f"Figures and tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
