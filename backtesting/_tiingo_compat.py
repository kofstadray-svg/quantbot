"""
backtesting/_tiingo_compat.py -- Drop-in replacement for yf.download()
                                  using Alpaca as the primary data source.

Usage in any backtesting script:
    # replace:  import yfinance as yf
    # with:     from backtesting._tiingo_compat import download as _yf_dl
    # then:     yf.download(...) -> _yf_dl(...)

Returns identical column structure to yf.download() after the standard
MultiIndex-flatten that every backtesting script does:
    Uppercase single-level columns: Open, High, Low, Close, Volume

Fallback chain:
    1. Alpaca Market Data API  (same ALPACA_API_KEY already in .env)
    2. Tiingo REST API         (TIINGO_API_KEY in .env — secondary)
    3. yfinance                (free, no auth — final fallback)
"""
from __future__ import annotations
import pandas as pd
from loguru import logger


def download(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    *,
    progress: bool = False,
    auto_adjust: bool = True,
    timeout: int = 20,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch OHLCV via Alpaca (primary), Tiingo (secondary), or yfinance (fallback).
    Returns flat-column DataFrame: Open, High, Low, Close, Volume.
    """
    _is_index = ticker.startswith("^")

    # --- 1. Alpaca (primary — no quota, same keys as broker) -----------------
    if not _is_index:
        try:
            from data.alpaca_data import (
                get_ohlcv as _alpaca_ohlcv,
                get_crypto_ohlcv as _alpaca_crypto,
            )
            _is_crypto = "-USD" in ticker.upper()
            if _is_crypto:
                df = _alpaca_crypto(ticker, period=period, interval=interval)
            else:
                df = _alpaca_ohlcv(ticker, period=period, interval=interval)

            if df is not None and not df.empty:
                rename = {
                    "open":  "Open", "high": "High",
                    "low":   "Low",  "close": "Close", "volume": "Volume",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                return df
        except Exception as exc:
            logger.debug("_tiingo_compat | {} Alpaca failed ({}), trying Tiingo", ticker, exc)

    # --- 2. Tiingo (secondary) -----------------------------------------------
    if not _is_index:
        try:
            from config import TIINGO_ENABLED
            if TIINGO_ENABLED:
                _is_crypto = "-USD" in ticker.upper()
                if _is_crypto:
                    from data.tiingo_data import get_crypto_ohlcv
                    df = get_crypto_ohlcv(ticker, period=period, interval=interval)
                else:
                    from data.tiingo_data import get_ohlcv
                    df = get_ohlcv(ticker, period=period, interval=interval)

                if df is not None and not df.empty:
                    rename = {
                        "open":      "Open",  "high":  "High",
                        "low":       "Low",   "close": "Close",
                        "volume":    "Volume", "adj_close": "Adj Close",
                    }
                    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                    return df
        except Exception as exc:
            logger.debug("_tiingo_compat | {} Tiingo failed ({}), using yfinance", ticker, exc)

    # --- 3. yfinance (final fallback) ----------------------------------------
    try:
        import yfinance as yf
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=progress, auto_adjust=auto_adjust, timeout=timeout,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        logger.warning("_tiingo_compat | {} yfinance failed: {}", ticker, exc)
        return pd.DataFrame()
