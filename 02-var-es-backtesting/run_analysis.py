"""VaR and Expected Shortfall backtesting for an equity portfolio.

Estimates 1-day VaR and ES for an equal-weighted five-stock portfolio
with three methods (historical, parametric, Monte Carlo), then — the
real point — backtests each method out of sample with a 250-day rolling
window and judges it with the Kupiec, Christoffersen and Basel traffic
light criteria.

Outputs (written to outputs/):
- return_distribution.png: portfolio returns with VaR/ES marked
- backtest_exceptions.png: losses vs rolling VaR, exceptions in red
- backtest_results.csv: exception counts, test p-values, Basel zones
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting import (
    BacktestResult,
    basel_traffic_light_zone,
    christoffersen_independence_test,
    kupiec_pof_test,
    rolling_backtest,
)
from src.data_loader import TICKERS, compute_daily_returns, load_prices
from src.var_models import (
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
)

METHOD_NAMES = ["historical", "parametric", "monte_carlo"]
CONFIDENCE_LEVELS = (0.95, 0.99)
WINDOW_DAYS = 250  # one trading year, the Basel estimation window
N_MC_SIMULATIONS = 5_000
RANDOM_SEED = 42

OUTPUT_DIR = Path(__file__).parent / "outputs"


def plot_return_distribution(
    portfolio_returns: np.ndarray,
    asset_returns: pd.DataFrame,
    portfolio_weights: np.ndarray,
) -> None:
    """Plot the return histogram with full-sample 99% VaR and ES marked.

    Args:
        portfolio_returns: daily portfolio returns over the full sample.
        asset_returns: daily asset returns (for the Monte Carlo method).
        portfolio_weights: weight per asset, summing to 1.
    """
    estimates = {
        "historical": historical_var_es(portfolio_returns, 0.99),
        "parametric": parametric_var_es(portfolio_returns, 0.99),
        "monte_carlo": monte_carlo_var_es(
            asset_returns.to_numpy(),
            portfolio_weights,
            0.99,
            N_MC_SIMULATIONS,
            RANDOM_SEED,
        ),
    }
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.hist(
        100 * portfolio_returns, bins=100, color="steelblue", alpha=0.7
    )
    colours = {"historical": "red", "parametric": "orange", "monte_carlo": "green"}
    for method_name, estimate in estimates.items():
        # VaR and ES are positive losses, so they sit on the negative
        # (loss) side of the return axis.
        axis.axvline(
            -100 * estimate.value_at_risk,
            color=colours[method_name],
            linewidth=1.8,
            label=f"{method_name} VaR 99% = {100 * estimate.value_at_risk:.2f}%",
        )
        axis.axvline(
            -100 * estimate.expected_shortfall,
            color=colours[method_name],
            linewidth=1.8,
            linestyle="--",
            label=f"{method_name} ES 99% = {100 * estimate.expected_shortfall:.2f}%",
        )
    axis.set_xlabel("Daily portfolio return (%)")
    axis.set_ylabel("Number of days")
    axis.set_title("Portfolio return distribution with 99% VaR (solid) and ES (dashed)")
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "return_distribution.png", dpi=150)
    plt.close(figure)


def plot_backtest_exceptions(
    backtest_results: dict[str, BacktestResult],
) -> None:
    """Plot daily losses against each method's rolling 99% VaR forecast.

    Exception days — where the realised loss pierced the forecast — are
    marked in red; visually clustered red dots are what the
    Christoffersen test measures formally.

    Args:
        backtest_results: 99% BacktestResult per method name.
    """
    figure, axes = plt.subplots(
        len(backtest_results), 1, figsize=(12, 10), sharex=True, sharey=True
    )
    for axis, (method_name, result) in zip(axes, backtest_results.items()):
        axis.plot(
            result.test_dates,
            100 * result.actual_losses,
            color="grey",
            linewidth=0.6,
            label="daily loss",
        )
        axis.plot(
            result.test_dates,
            100 * result.var_forecasts,
            color="black",
            linewidth=1.2,
            label="VaR 99% forecast",
        )
        exception_dates = result.test_dates[result.exception_flags]
        exception_losses = result.actual_losses[result.exception_flags]
        axis.scatter(
            exception_dates,
            100 * exception_losses,
            color="red",
            s=25,
            zorder=3,
            label=f"exceptions ({int(np.sum(result.exception_flags))})",
        )
        axis.set_title(f"{method_name} — rolling 250-day 99% VaR")
        axis.set_ylabel("Loss (%)")
        axis.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "backtest_exceptions.png", dpi=150)
    plt.close(figure)


def summarise_backtest(result: BacktestResult, method_name: str) -> dict:
    """Reduce one backtest to the row shown in the results table.

    Args:
        result: output of rolling_backtest.
        method_name: label for the table.

    Returns:
        Dict of headline statistics for one method/confidence pair.
    """
    n_test_days = len(result.exception_flags)
    n_exceptions = int(np.sum(result.exception_flags))
    kupiec = kupiec_pof_test(
        n_exceptions, n_test_days, result.confidence_level
    )
    christoffersen = christoffersen_independence_test(result.exception_flags)
    # Basel zones are defined only for 99% VaR over 250 days.
    if result.confidence_level == 0.99:
        basel_zone = basel_traffic_light_zone(result.exception_flags)
    else:
        basel_zone = "n/a"
    return {
        "method": method_name,
        "confidence": result.confidence_level,
        "test_days": n_test_days,
        "expected_exceptions": (1 - result.confidence_level) * n_test_days,
        "actual_exceptions": n_exceptions,
        "kupiec_p_value": kupiec.p_value,
        "christoffersen_p_value": christoffersen.p_value,
        "basel_zone_last_250d": basel_zone,
    }


def main() -> None:
    """Run the full VaR/ES backtesting analysis and write all outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    prices = load_prices()
    asset_returns = compute_daily_returns(prices)
    # Equal-weighted portfolio, rebalanced daily.
    portfolio_weights = np.full(len(TICKERS), 1.0 / len(TICKERS))
    portfolio_returns = asset_returns.to_numpy() @ portfolio_weights

    plot_return_distribution(
        portfolio_returns, asset_returns, portfolio_weights
    )

    summary_rows = []
    backtests_99 = {}
    for method_name in METHOD_NAMES:
        for confidence_level in CONFIDENCE_LEVELS:
            result = rolling_backtest(
                asset_returns,
                portfolio_weights,
                method_name,
                confidence_level,
                WINDOW_DAYS,
                N_MC_SIMULATIONS,
                RANDOM_SEED,
            )
            summary_rows.append(summarise_backtest(result, method_name))
            if confidence_level == 0.99:
                backtests_99[method_name] = result

    plot_backtest_exceptions(backtests_99)
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)

    print("=" * 78)
    print("VaR BACKTESTING - SUMMARY")
    print("=" * 78)
    print(f"Portfolio: equal-weighted {', '.join(TICKERS)}")
    print(f"Sample: {asset_returns.index[0].date()} to "
          f"{asset_returns.index[-1].date()} "
          f"({len(asset_returns)} days, {WINDOW_DAYS}-day rolling window)")
    print()
    print(summary_table.to_string(index=False, float_format="{:.4f}".format))
    print()
    print("Reading the table: a well calibrated model has actual close to")
    print("expected exceptions and BOTH p-values above 0.05. A small Kupiec")
    print("p-value means the exception COUNT is wrong; a small Christoffersen")
    print("p-value means exceptions CLUSTER (risk arrives in bursts).")
    print(f"Figures and tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
