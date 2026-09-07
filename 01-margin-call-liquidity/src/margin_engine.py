"""Variation and initial margin mechanics for a short futures hedge.

The trader is long physical commodity and short futures. The hedge is
neutral in value, but only the futures leg is marked to market daily:
when prices rise, the exchange demands cash (variation margin) on the
short futures today, while the offsetting gain on the physical inventory
is only realised when the commodity is sold. The gap between the two is
a pure liquidity (funding) risk, not a market risk.
"""

import numpy as np


def daily_variation_margin(
    futures_path: np.ndarray,
    n_contracts: int,
    contract_size: float,
) -> np.ndarray:
    """Compute the daily variation margin cash flow of the short hedge.

    Each day the exchange settles the change in the futures price in
    cash. A short futures position loses when the price rises, so the
    daily cash flow is minus the price change times the position size.
    Negative values are cash the trader must pay out.

    Args:
        futures_path: futures prices for one path, length n_days + 1.
        n_contracts: number of futures contracts sold short.
        contract_size: units of commodity per contract (e.g. barrels).

    Returns:
        Daily cash flows, length n_days (one per settlement day).
    """
    daily_price_changes = np.diff(futures_path)
    # Short futures position: pays out when the price rises, receives
    # cash when the price falls.
    return -n_contracts * contract_size * daily_price_changes


def cumulative_margin_balance(daily_margin_flows: np.ndarray) -> np.ndarray:
    """Compute the running total of variation margin paid and received.

    A negative balance means the trader has paid out more cash than it
    has received since the hedge was put on. This running balance, not
    any single day's flow, is what a credit line must cover.

    Args:
        daily_margin_flows: daily variation margin cash flows.

    Returns:
        Cumulative balance after each settlement day, same length.
    """
    return np.cumsum(daily_margin_flows)


def initial_margin_requirement(
    futures_price: float,
    n_contracts: int,
    contract_size: float,
    initial_margin_rate: float,
) -> float:
    """Compute the initial margin posted when the hedge is opened.

    Exchanges require collateral up front to cover potential losses
    before a defaulting position can be closed out. A flat percentage of
    the position's notional value is a simple proxy for the exchange's
    actual (SPAN-style) calculation.

    Args:
        futures_price: futures price when the position is opened.
        n_contracts: number of futures contracts sold short.
        contract_size: units of commodity per contract.
        initial_margin_rate: initial margin as a fraction of notional.

    Returns:
        Initial margin in currency units (a positive cash outflow).
    """
    position_notional = futures_price * n_contracts * contract_size
    return initial_margin_rate * position_notional


def peak_margin_outflow(
    cumulative_balance: np.ndarray,
    initial_margin: float,
) -> float:
    """Compute the worst cumulative funding need over the hedge horizon.

    The credit line must cover the initial margin plus the deepest
    trough of the cumulative variation margin balance. If the balance
    never goes negative (prices fell, the short hedge received cash),
    only the initial margin is needed.

    Args:
        cumulative_balance: running variation margin balance over time.
        initial_margin: initial margin posted at inception.

    Returns:
        Peak cash outflow in currency units (a positive number).
    """
    worst_variation_loss = max(0.0, float(np.max(-cumulative_balance)))
    return initial_margin + worst_variation_loss
