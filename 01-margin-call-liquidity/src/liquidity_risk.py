"""Liquidity-at-Risk from the simulated distribution of margin outflows.

Liquidity-at-Risk (LaR) answers the treasurer's question: "how much
committed credit do we need so that, with a given confidence, we can
meet every margin call over the hedge horizon without a fire sale?"
It is the analogue of Value-at-Risk, but on cumulative cash outflows
instead of portfolio value.
"""

import numpy as np

from src.margin_engine import (
    cumulative_margin_balance,
    daily_variation_margin,
    initial_margin_requirement,
    peak_margin_outflow,
)


def simulate_peak_outflows(
    futures_paths: np.ndarray,
    n_contracts: int,
    contract_size: float,
    initial_margin_rate: float,
) -> np.ndarray:
    """Compute the peak margin outflow on every simulated path.

    Each path is one possible future: the margin engine turns it into
    daily cash flows and records the worst cumulative funding need. The
    collection over all paths is the distribution that LaR is read from.

    Args:
        futures_paths: futures prices, shape (n_paths, n_days + 1).
        n_contracts: number of futures contracts sold short.
        contract_size: units of commodity per contract.
        initial_margin_rate: initial margin as a fraction of notional.

    Returns:
        Peak outflow per path, length n_paths (positive numbers).
    """
    n_paths = futures_paths.shape[0]
    peak_outflows = np.zeros(n_paths)
    for path_index in range(n_paths):
        futures_path = futures_paths[path_index]
        margin_flows = daily_variation_margin(
            futures_path, n_contracts, contract_size
        )
        margin_balance = cumulative_margin_balance(margin_flows)
        initial_margin = initial_margin_requirement(
            futures_path[0], n_contracts, contract_size, initial_margin_rate
        )
        peak_outflows[path_index] = peak_margin_outflow(
            margin_balance, initial_margin
        )
    return peak_outflows


def liquidity_at_risk(
    peak_outflows: np.ndarray,
    confidence_level: float,
) -> float:
    """Compute Liquidity-at-Risk at a given confidence level.

    LaR at 99% is the size of credit line that would have been enough
    to meet every margin call in 99% of the simulated scenarios.

    Args:
        peak_outflows: peak margin outflow per simulated path.
        confidence_level: e.g. 0.95 or 0.99.

    Returns:
        LaR in currency units.
    """
    return float(np.percentile(peak_outflows, 100.0 * confidence_level))
