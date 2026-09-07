"""Implied volatility: invert Black-Scholes for the market's volatility.

The market quotes prices; the model's only free parameter is the
volatility. The implied volatility is the sigma at which Black-Scholes
reproduces the quoted price. Because the Black-Scholes price is
strictly increasing in volatility (vega > 0), the inversion has at most
one solution and a bracketing root finder is guaranteed to find it —
provided the quote sits inside the model's attainable price range.
"""

from scipy.optimize import brentq

from src.black_scholes import option_price

# Search bracket: from essentially-zero to 500% volatility, wide enough
# for any realistic market quote.
MIN_VOLATILITY = 1e-6
MAX_VOLATILITY = 5.0


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Solve for the volatility that reproduces a quoted option price.

    Uses Brent's method, a bracketing root finder: it needs a low and a
    high volatility whose model prices straddle the quote, and then
    converges without derivatives. Before solving, the quote is checked
    against the price range attainable within the bracket; a quote
    outside that range (below intrinsic value, or above the asset
    price) admits no implied volatility, which signals a stale or
    arbitrageable quote rather than a numerical problem.

    Args:
        market_price: quoted option price.
        spot: current asset price.
        strike: option strike.
        time_to_expiry: time to expiry in years.
        risk_free_rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield q.
        option_type: "call" or "put".

    Returns:
        Implied volatility (annualised).

    Raises:
        ValueError: if no volatility in the bracket reproduces the quote.
    """
    price_at_min_vol = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        MIN_VOLATILITY, option_type,
    )
    price_at_max_vol = option_price(
        spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
        MAX_VOLATILITY, option_type,
    )
    if not price_at_min_vol <= market_price <= price_at_max_vol:
        raise ValueError(
            f"No implied volatility exists for {option_type} quote "
            f"{market_price:.4f}: attainable model prices are "
            f"[{price_at_min_vol:.4f}, {price_at_max_vol:.4f}]. The quote "
            "violates no-arbitrage bounds (stale or bad price?)."
        )

    def pricing_error(volatility: float) -> float:
        """Model price minus market quote at a trial volatility."""
        model_price = option_price(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
            volatility, option_type,
        )
        return model_price - market_price

    return float(
        brentq(pricing_error, MIN_VOLATILITY, MAX_VOLATILITY, xtol=1e-12)
    )
