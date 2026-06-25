"""
data/alpaca_data.py — Alpaca Market Data API wrapper.

Replaces Tiingo as the primary OHLCV data source.  Uses the same
ALPACA_API_KEY / ALPACA_SECRET_KEY already in .env for order execution —
no new credentials needed.

Advantages over Tiingo:
  • No daily request quota — designed for algo trading bots
  • Same API keys already in use (zero new setup)
  • No circuit breaker needed — generous rate limits
  • Serves all US equities and crypto in the bot's watchlists
  • 30-minute in-memory session cache (same as Tiingo)

Limitations vs Tiingo:
  • Free IEX feed: US equities only, market hours only
  • No index data (^VIX, ^GSPC) — yfinance handles those
  • Crypto symbols use BASE/QUOTE format (BTC/USD not BTC-USD)
    → handled transparently by _normalize_crypto_symbol()

Data feed:
  Stocks : IEX (free, included with paper + live accounts)
  Crypto : Alpaca crypto feed (no auth required)

Function signatures mirror tiingo_data.py so market_data.py needs
only a one-line import change.
"""
from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Clients (lazy-initialised, one per process)
# ---------------------------------------------------------------------------

_STOCK_CLIENT  = None
_CRYPTO_CLIENT = None
_CLIENT_LOCK   = threading.Lock()


def _get_stock_client():
    global _STOCK_CLIENT
    if _STOCK_CLIENT is not None:
        return _STOCK_CLIENT
    with _CLIENT_LOCK:
        if _STOCK_CLIENT is not None:
            return _STOCK_CLIENT
        from alpaca.data.historical import StockHistoricalDataClient
        # Ensure .env is loaded before reading keys
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key:
            try:
                from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
                key, secret = ALPACA_API_KEY, ALPACA_SECRET_KEY
            except ImportError:
                pass
        _STOCK_CLIENT = StockHistoricalDataClient(api_key=key, secret_key=secret)
        logger.info("Alpaca data | StockHistoricalDataClient ready (key: ...{})", key[-4:] if key else "none")
        return _STOCK_CLIENT


def _get_crypto_client():
    global _CRYPTO_CLIENT
    if _CRYPTO_CLIENT is not None:
        return _CRYPTO_CLIENT
    with _CLIENT_LOCK:
        if _CRYPTO_CLIENT is not None:
            return _CRYPTO_CLIENT
        from alpaca.data.historical import CryptoHistoricalDataClient
        _CRYPTO_CLIENT = CryptoHistoricalDataClient()   # no auth for crypto
        logger.info("Alpaca data | CryptoHistoricalDataClient ready")
        return _CRYPTO_CLIENT


# ---------------------------------------------------------------------------
# In-memory session cache (30-min TTL)
# ---------------------------------------------------------------------------

_MEM_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_MEM_CACHE_LOCK = threading.Lock()
_MEM_CACHE_TTL_SEC = 1800   # 30 minutes


def _mem_get(key: str) -> pd.DataFrame | None:
    with _MEM_CACHE_LOCK:
        entry = _MEM_CACHE.get(key)
    if entry and (time.time() - entry[0]) < _MEM_CACHE_TTL_SEC:
        return entry[1].copy()
    return None


def _mem_put(key: str, df: pd.DataFrame) -> None:
    with _MEM_CACHE_LOCK:
        _MEM_CACHE[key] = (time.time(), df.copy())


# ---------------------------------------------------------------------------
# Period → datetime conversion
# ---------------------------------------------------------------------------

def _period_to_start(period: str) -> datetime:
    """Convert yfinance-style period string to a UTC datetime."""
    now = datetime.now(timezone.utc)
    mapping = {
        "1d":  timedelta(days=1),
        "5d":  timedelta(days=5),
        "1mo": timedelta(days=31),
        "2mo": timedelta(days=62),
        "3mo": timedelta(days=92),
        "6mo": timedelta(days=183),
        "1y":  timedelta(days=365),
        "2y":  timedelta(days=730),
        "5y":  timedelta(days=1826),
    }
    return now - mapping.get(period, timedelta(days=365))


# ---------------------------------------------------------------------------
# TimeFrame helper
# ---------------------------------------------------------------------------

def _interval_to_timeframe(interval: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    mapping = {
        "1d":  TimeFrame.Day,
        "1h":  TimeFrame.Hour,
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
        "1m":  TimeFrame.Minute,
    }
    return mapping.get(interval, TimeFrame.Day)


# ---------------------------------------------------------------------------
# Normalise Alpaca BarSet response → clean DataFrame
# ---------------------------------------------------------------------------

def _bars_to_df(bars, symbol: str) -> pd.DataFrame | None:
    """Convert Alpaca BarSet/DataFrame to a clean lowercase-column DataFrame."""
    try:
        df = bars.df
        if df is None or df.empty:
            return None

        # Multi-index (symbol, timestamp) → filter + flatten
        if isinstance(df.index, pd.MultiIndex):
            if symbol.upper() in df.index.get_level_values(0):
                df = df.xs(symbol.upper(), level=0)
            else:
                # Try without /USD suffix for crypto
                base = symbol.upper().replace("/USD", "").replace("-USD", "")
                matches = [s for s in df.index.get_level_values(0) if base in s]
                if matches:
                    df = df.xs(matches[0], level=0)
                else:
                    return None

        df = df.copy()
        df.index.name = "date"

        # Rename Alpaca columns to lowercase OHLCV
        rename_map = {
            "open":         "open",
            "high":         "high",
            "low":          "low",
            "close":        "close",
            "volume":       "volume",
            "trade_count":  "trade_count",
            "vwap":         "vwap",
        }
        df.columns = [c.lower() for c in df.columns]
        df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]

        # Drop incomplete bars (pre-market partial bars have NaN high/low)
        ohlc_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = df.dropna(subset=ohlc_cols)

        if df.empty:
            return None

        return df

    except Exception as exc:
        logger.debug("Alpaca data | _bars_to_df failed for {}: {}", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Equity OHLCV
# ---------------------------------------------------------------------------

def get_ohlcv(
    ticker:   str,
    period:   str = "1y",
    interval: str = "1d",
    *,
    start: datetime | None = None,
    end:   datetime | None = None,
) -> pd.DataFrame | None:
    """
    Fetch daily (or intraday) OHLCV for a US equity via Alpaca IEX feed.

    Returns a DataFrame with lowercase columns: open, high, low, close, volume.
    Returns None on any error (caller should fall back to yfinance).

    Args:
        ticker   : stock symbol (e.g. "AAPL", "SPY")
        period   : yfinance-style lookback ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y")
        interval : bar size ("1d", "1h", "5m", "1m")
        start    : explicit start datetime (overrides period)
        end      : explicit end datetime (defaults to now)
    """
    # Skip index tickers — Alpaca doesn't serve ^VIX, ^GSPC etc.
    if ticker.startswith("^") or ticker in ("VIX", "VIXY"):
        return None

    end_dt   = end   or datetime.now(timezone.utc)
    start_dt = start or _period_to_start(period)

    mem_key  = f"eq:{ticker.upper()}:{period}:{interval}"
    cached   = _mem_get(mem_key)
    if cached is not None:
        logger.debug("Alpaca data | {} [mem-cache hit]", ticker)
        return cached

    try:
        from alpaca.data.requests import StockBarsRequest
        client  = _get_stock_client()
        tf      = _interval_to_timeframe(interval)
        request = StockBarsRequest(
            symbol_or_symbols = ticker.upper(),
            timeframe         = tf,
            start             = start_dt,
            end               = end_dt,
            feed              = "iex",        # free feed (paper + live accounts)
            adjustment        = "all",        # split + dividend adjusted
        )
        bars   = client.get_stock_bars(request)
        df     = _bars_to_df(bars, ticker)

        if df is not None and not df.empty:
            _mem_put(mem_key, df)
            logger.debug("Alpaca data | {} {} bars ({})", ticker, len(df), period)
            return df

        logger.debug("Alpaca data | {} returned empty", ticker)
        return None

    except Exception as exc:
        logger.warning("Alpaca data | {} fetch error: {}", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Crypto OHLCV
# ---------------------------------------------------------------------------

def _yf_to_alpaca_crypto(symbol: str) -> str:
    """Convert BTC-USD (yfinance) to BTC/USD (Alpaca)."""
    s = symbol.upper()
    if "-" in s:
        base, quote = s.split("-", 1)
        return f"{base}/{quote}"
    return s


def get_crypto_ohlcv(
    symbol:   str,
    period:   str = "6mo",
    interval: str = "1d",
    *,
    start: datetime | None = None,
    end:   datetime | None = None,
) -> pd.DataFrame | None:
    """
    Fetch daily crypto OHLCV via Alpaca crypto feed.

    Accepts BTC-USD (yfinance format) or BTC/USD (Alpaca format).
    Returns lowercase-column DataFrame or None on error.
    """
    alpaca_symbol = _yf_to_alpaca_crypto(symbol)
    end_dt        = end   or datetime.now(timezone.utc)
    start_dt      = start or _period_to_start(period)

    mem_key = f"cr:{alpaca_symbol}:{period}:{interval}"
    cached  = _mem_get(mem_key)
    if cached is not None:
        logger.debug("Alpaca data | crypto {} [mem-cache hit]", symbol)
        return cached

    try:
        from alpaca.data.requests import CryptoBarsRequest
        client  = _get_crypto_client()
        tf      = _interval_to_timeframe(interval)
        request = CryptoBarsRequest(
            symbol_or_symbols = alpaca_symbol,
            timeframe         = tf,
            start             = start_dt,
            end               = end_dt,
        )
        bars = client.get_crypto_bars(request)
        df   = _bars_to_df(bars, alpaca_symbol)

        if df is not None and not df.empty:
            _mem_put(mem_key, df)
            logger.debug("Alpaca data | crypto {} {} bars ({})", symbol, len(df), period)
            return df

        logger.debug("Alpaca data | crypto {} returned empty", symbol)
        return None

    except Exception as exc:
        logger.warning("Alpaca data | crypto {} fetch error: {}", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Batch equity OHLCV (used by RS ranking — fetches ~600 tickers)
# ---------------------------------------------------------------------------

def get_batch_ohlcv(
    tickers:  list[str],
    period:   str = "2mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily bars for multiple tickers in a single API call.
    Alpaca handles batching natively — much faster than Tiingo's per-ticker calls.
    Returns {ticker: DataFrame}.
    """
    if not tickers:
        return {}

    end_dt   = datetime.now(timezone.utc)
    start_dt = _period_to_start(period)

    try:
        from alpaca.data.requests import StockBarsRequest
        client  = _get_stock_client()
        tf      = _interval_to_timeframe(interval)
        request = StockBarsRequest(
            symbol_or_symbols = [t.upper() for t in tickers],
            timeframe         = tf,
            start             = start_dt,
            end               = end_dt,
            feed              = "iex",
            adjustment        = "all",
        )
        bars_set = client.get_stock_bars(request)
        df_multi = bars_set.df

        result: dict[str, pd.DataFrame] = {}
        if df_multi is None or df_multi.empty:
            return result

        # MultiIndex: (symbol, timestamp)
        for sym in df_multi.index.get_level_values(0).unique():
            df = df_multi.xs(sym, level=0).copy()
            df.columns = [c.lower() for c in df.columns]
            df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
            df = df.dropna(subset=[c for c in ("open", "high", "low", "close") if c in df.columns])
            if not df.empty:
                result[sym] = df

        logger.info("Alpaca data | batch {} tickers → {} returned", len(tickers), len(result))
        return result

    except Exception as exc:
        logger.warning("Alpaca data | batch fetch error: {}", exc)
        return {}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG")

    print("\n=== Alpaca data self-test ===\n")

    print("--- AAPL daily 1y ---")
    df = get_ohlcv("AAPL", period="1y")
    if df is not None:
        print(f"  {len(df)} bars  last close: ${df['close'].iloc[-1]:.2f}  cols: {list(df.columns)}")
    else:
        print("  FAILED")

    print("\n--- SPY daily 3mo ---")
    df = get_ohlcv("SPY", period="3mo")
    if df is not None:
        print(f"  {len(df)} bars  last close: ${df['close'].iloc[-1]:.2f}")
    else:
        print("  FAILED")

    print("\n--- BTC-USD daily 6mo ---")
    df = get_crypto_ohlcv("BTC-USD", period="6mo")
    if df is not None:
        print(f"  {len(df)} bars  last close: ${df['close'].iloc[-1]:,.0f}")
    else:
        print("  FAILED")

    print("\n--- Batch NVDA+PLTR+CRWD 2mo ---")
    batch = get_batch_ohlcv(["NVDA", "PLTR", "CRWD"], period="2mo")
    for sym, df in batch.items():
        print(f"  {sym}: {len(df)} bars  last: ${df['close'].iloc[-1]:.2f}")

    print("\n=== done ===\n")
