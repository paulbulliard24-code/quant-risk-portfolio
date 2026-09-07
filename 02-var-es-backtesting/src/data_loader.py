"""Price data loading for a small equity portfolio.

The portfolio is five large-cap US stocks from different sectors, so the
covariance matrix used later is neither trivial (one asset) nor
degenerate (many near-identical assets).

Prices are downloaded once from Yahoo Finance and cached to
data/prices.csv, and that CSV is committed to the repository. This keeps
every result reproducible with no network access: the download only
happens if the cache file is missing.
"""

from pathlib import Path

import pandas as pd

# Five sectors: technology, banking, energy, consumer staples, health care.
TICKERS = ["AAPL", "JPM", "XOM", "PG", "JNJ"]

# Fixed dates (about five years of daily data) so a fresh download and
# the committed cache cover the same sample.
START_DATE = "2021-09-01"
END_DATE = "2026-09-01"

CACHE_PATH = Path(__file__).parent.parent / "data" / "prices.csv"


def download_prices(
    tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance.

    Adjusted closes are used so that dividends and splits do not appear
    as spurious price jumps in the return series.

    Args:
        tickers: stock symbols to download.
        start_date: first date, "YYYY-MM-DD".
        end_date: last date (exclusive), "YYYY-MM-DD".

    Returns:
        DataFrame of prices, one column per ticker, indexed by date.
    """
    # Imported here so the cached-data path works without yfinance installed.
    import yfinance

    raw_data = yfinance.download(
        tickers, start=start_date, end=end_date, auto_adjust=True
    )
    # yfinance returns one column block per field; keep the close prices.
    prices = raw_data["Close"][tickers]
    return prices.dropna()


def load_prices() -> pd.DataFrame:
    """Load portfolio prices from the local cache, downloading if absent.

    Returns:
        DataFrame of daily adjusted close prices, one column per ticker.
    """
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    prices.to_csv(CACHE_PATH)
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a price series.

    Simple (not log) returns are used because portfolio return is then
    exactly the weighted average of asset returns, which is what the
    variance-covariance and Monte Carlo methods aggregate.

    Args:
        prices: daily prices, one column per asset.

    Returns:
        Daily returns with the first (undefined) day dropped.
    """
    return prices.pct_change().dropna()
