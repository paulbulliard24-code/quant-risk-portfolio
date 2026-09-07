"""Closed-form Black-Scholes-Merton pricing and analytical Greeks.

European options on an asset paying a continuous dividend yield q
(Merton's extension; set q = 0 for a non-dividend stock).

Unit conventions — every Greek here is returned in the units a trading
desk quotes, not the raw calculus derivative:
- delta, gamma: per $1 move in spot (raw derivative, standard quote)
- vega: per 1 volatility POINT, e.g. 25% -> 26% (raw derivative / 100)
- theta: per 1 CALENDAR DAY of time decay (raw per-year theta / 365)
- rho: per 1 BASIS POINT of rate move (raw derivative / 10,000)
- vanna: change of delta per 1 volatility point (raw / 100)
- volga: change of desk vega per 1 volatility point (raw / 10,000)
"""

import numpy as np
from scipy import stats


def compute_d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float]:
    """Compute the two standardised log-moneyness terms d1 and d2.

    d1 is the (risk-neutral, volatility-adjusted) standardised distance
    of the forward from the strike; N(d2) is the risk-neutral
    probability that the option finishes in the money.

    Args:
        spot: current asset price.
        strike: option strike.
        time_to_expiry: time to expiry in years (must be > 0).
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        volatility: annualised volatility (must be > 0).

    Returns:
        (d1, d2).
    """
    volatility_over_expiry = volatility * np.sqrt(time_to_expiry)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2)
        * time_to_expiry
    ) / volatility_over_expiry
    d2 = d1 - volatility_over_expiry
    return d1, d2


def option_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    """Black-Scholes-Merton price of a European call or put.

    The price is the discounted risk-neutral expectation of the payoff:
    the call is a long position in e^(-qT) N(d1) units of the asset
    financed by borrowing K e^(-rT) N(d2).

    Args:
        spot: current asset price.
        strike: option strike.
        time_to_expiry: time to expiry in years.
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        volatility: annualised volatility.
        option_type: "call" or "put".

    Returns:
        Option price in currency units.
    """
    d1, d2 = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-risk_free_rate * time_to_expiry)
    if option_type == "call":
        return discounted_spot * stats.norm.cdf(d1) - discounted_strike * stats.norm.cdf(d2)
    if option_type == "put":
        return discounted_strike * stats.norm.cdf(-d2) - discounted_spot * stats.norm.cdf(-d1)
    raise ValueError(f"Unknown option type: {option_type}")


def delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    """Delta: price change per $1 move in spot (the hedge ratio).

    A call delta lies in (0, e^(-qT)); the put delta is the call delta
    minus e^(-qT) (differentiate put-call parity with respect to spot).

    Args: as in option_price.

    Returns:
        Delta per $1 of spot.
    """
    d1, _ = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    if option_type == "call":
        return dividend_discount * stats.norm.cdf(d1)
    if option_type == "put":
        return dividend_discount * (stats.norm.cdf(d1) - 1.0)
    raise ValueError(f"Unknown option type: {option_type}")


def gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Gamma: change of delta per $1 move in spot.

    Identical for a call and a put with the same inputs, because the
    two prices differ only by the parity terms, which are linear in
    spot and so have zero second derivative.

    Args: as in option_price (no option_type needed).

    Returns:
        Gamma per $1 of spot.
    """
    d1, _ = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    return (
        dividend_discount
        * stats.norm.pdf(d1)
        / (spot * volatility * np.sqrt(time_to_expiry))
    )


def vega_per_vol_point(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Vega: price change per 1 volatility point (e.g. 25% -> 26%).

    Market convention: the raw derivative dPrice/dSigma is divided by
    100 so the number matches a one-point quote move. Identical for
    calls and puts (parity terms do not depend on volatility).

    Args: as in option_price (no option_type needed).

    Returns:
        Vega in currency units per volatility point.
    """
    d1, _ = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    raw_vega = (
        spot * dividend_discount * stats.norm.pdf(d1) * np.sqrt(time_to_expiry)
    )
    return raw_vega / 100.0


def theta_per_day(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    """Theta: price change per 1 calendar day of time decay.

    Market convention: the raw per-year derivative is divided by 365 so
    the number answers "what does holding this overnight cost?". The
    three terms are: loss of optionality (always negative), interest on
    the strike leg, and the dividend earned or forgone on the spot leg.

    Args: as in option_price.

    Returns:
        Theta in currency units per calendar day (usually negative).
    """
    d1, d2 = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    rate_discount = np.exp(-risk_free_rate * time_to_expiry)
    optionality_decay = -(
        spot * dividend_discount * stats.norm.pdf(d1) * volatility
    ) / (2.0 * np.sqrt(time_to_expiry))
    if option_type == "call":
        theta_per_year = (
            optionality_decay
            - risk_free_rate * strike * rate_discount * stats.norm.cdf(d2)
            + dividend_yield * spot * dividend_discount * stats.norm.cdf(d1)
        )
    elif option_type == "put":
        theta_per_year = (
            optionality_decay
            + risk_free_rate * strike * rate_discount * stats.norm.cdf(-d2)
            - dividend_yield * spot * dividend_discount * stats.norm.cdf(-d1)
        )
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    return theta_per_year / 365.0


def rho_per_basis_point(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    """Rho: price change per 1 basis point move in the risk-free rate.

    Market convention: the raw derivative dPrice/dRate is divided by
    10,000. Only the discounted strike leg depends on the rate, so a
    call gains when rates rise (cheaper to defer paying the strike) and
    a put loses.

    Args: as in option_price.

    Returns:
        Rho in currency units per basis point.
    """
    _, d2 = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    rate_discount = np.exp(-risk_free_rate * time_to_expiry)
    if option_type == "call":
        raw_rho = strike * time_to_expiry * rate_discount * stats.norm.cdf(d2)
    elif option_type == "put":
        raw_rho = -strike * time_to_expiry * rate_discount * stats.norm.cdf(-d2)
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    return raw_rho / 10_000.0


def vanna_per_vol_point(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Vanna: change of delta per 1 volatility point.

    The cross-derivative d2Price / (dSpot dSigma), divided by 100 for
    the one-point convention. It tells a desk how its spot hedge drifts
    when implied volatility moves — the key risk of skew positions.
    Identical for calls and puts.

    Args: as in option_price (no option_type needed).

    Returns:
        Delta change per volatility point.
    """
    d1, d2 = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    raw_vanna = -dividend_discount * stats.norm.pdf(d1) * d2 / volatility
    return raw_vanna / 100.0


def volga_per_vol_point(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Volga: change of (desk) vega per 1 volatility point.

    The second derivative d2Price / dSigma2, divided by 10,000 because
    both differentiations are converted to one-point moves. Positive
    volga means the position gains vega as volatility rises — it is
    long convexity in volatility. Identical for calls and puts.

    Args: as in option_price (no option_type needed).

    Returns:
        Change of vega-per-vol-point, per volatility point.
    """
    d1, d2 = compute_d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
    )
    dividend_discount = np.exp(-dividend_yield * time_to_expiry)
    raw_vega = (
        spot * dividend_discount * stats.norm.pdf(d1) * np.sqrt(time_to_expiry)
    )
    raw_volga = raw_vega * d1 * d2 / volatility
    return raw_volga / 10_000.0
