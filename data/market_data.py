"""
data/market_data.py - Fetch OHLCV and quote data.

Primary source:   Alpaca Market Data API (same API keys as broker, no quota).
Secondary source: Tiingo REST API (when TIINGO_API_KEY is set in .env).
Fallback:         yfinance (free, no auth, rate-limited by Yahoo).

Priority chain:
  1. Alpaca  — no daily request quota, designed for algo trading bots
  2. Tiingo  — cleaner adjusted prices when Alpaca returns nothing
  3. yfinance — free fallback, used when both primary sources fail

To activate Tiingo secondary: set TIINGO_API_KEY in .env.
"""
from __future__ import annotations
import concurrent.futures as _futures
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from loguru import logger
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, TIINGO_ENABLED
from utils.retry import with_retry
from utils import cache as _cache

# Tiingo — imported lazily so the module loads fine when key is not yet set
try:
    from data.tiingo_data import (
        get_ohlcv      as _tiingo_ohlcv,
        get_crypto_ohlcv as _tiingo_crypto_ohlcv,
        TiingoDisabledError,
    )
    _TIINGO_AVAILABLE = True
except ImportError:
    _TIINGO_AVAILABLE = False
    TiingoDisabledError = Exception  # type: ignore[assignment,misc]

_alpaca_data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# How stale a cached OHLCV fallback may be before we refuse it
_OHLCV_CACHE_MAX_AGE_SEC = 4 * 24 * 3600

# ---------------------------------------------------------------------------
# yfinance helpers — kept as fallback when Tiingo key is absent
# ---------------------------------------------------------------------------

_YF_TIMEOUT      = 20
_YF_WALL_TIMEOUT = 45


def _run_with_wall_timeout(fn, timeout: float, ticker: str):
    """Run fn() in a thread; raise ConnectionError if it exceeds wall-clock."""
    with _futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except _futures.TimeoutError:
            raise ConnectionError(
                f"yfinance download for {ticker} exceeded wall-clock limit "
                f"({timeout:.0f}s)"
            )


@with_retry(max_attempts=3, base_delay=3.0)
def _get_ohlcv_yfinance(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance (fallback path)."""
    logger.debug("yfinance fallback | {} period={} interval={}", ticker, period, interval)

    def _download():
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, timeout=_YF_TIMEOUT)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df

    return _run_with_wall_timeout(_download, _YF_WALL_TIMEOUT, ticker)


# ---------------------------------------------------------------------------
# Public API — Tiingo first, yfinance fallback
# ---------------------------------------------------------------------------


def get_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV with Tiingo-first routing and last-good-data fallback.

    Priority:
      1. Tiingo REST API (when TIINGO_API_KEY is set)
      2. yfinance (when Tiingo key is absent or Tiingo fails)
      3. Disk cache (when both live sources fail, up to 4 days old)
    """
    cache_key = f"ohlcv_{ticker}_{period}_{interval}"

    def _fetch_live() -> pd.DataFrame:
        # --- Alpaca (primary — no quota, same keys as broker) ---
        try:
            from data.alpaca_data import get_ohlcv as _alpaca_ohlcv
            df = _alpaca_ohlcv(ticker, period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception as _ae:
            logger.debug("market_data | {} Alpaca failed ({}), trying Tiingo", ticker, _ae)
        # --- Tiingo (secondary) ---
        if _TIINGO_AVAILABLE and TIINGO_ENABLED:
            try:
                df = _tiingo_ohlcv(ticker, period=period, interval=interval)
                if df is not None and not df.empty:
                    return df
                logger.warning("Tiingo | {} returned empty, trying yfinance", ticker)
            except TiingoDisabledError:
                pass
            except Exception as exc:
                logger.warning("Tiingo | {} fetch error ({}), falling back to yfinance", ticker, exc)
        # --- yfinance (tertiary fallback) ---
        return _get_ohlcv_yfinance(ticker, period, interval)

    try:
        df = _fetch_live()
        if df is not None and not df.empty:
            # Drop incomplete bars (pre-market yfinance includes today's partial
            # bar with NaN high/low; those NaNs propagate to ADX/ATR/slope in
            # market_regime and produce ADX=nan, ATR_exp=nan, slope=nan which
            # misclassifies regime and corrupts threshold selection).
            ohlc_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
            df = df.dropna(subset=ohlc_cols)
            if not df.empty:
                try:
                    _cache.save(cache_key,
                                {"json": df.to_json(orient="split", date_format="iso")},
                                label=f"OHLCV {ticker} {period}/{interval}")
                except Exception as e:
                    logger.debug("OHLCV cache save failed for {}: {}", ticker, e)
        return df
    except Exception as live_exc:
        entry = _cache.load(cache_key, max_age_sec=_OHLCV_CACHE_MAX_AGE_SEC)
        if entry is not None:
            age_min = entry["age_sec"] / 60.0
            logger.warning(
                "OHLCV LIVE FETCH FAILED for {} ({}); "
                "serving CACHED fallback ({:.0f} min old). "
                "Screening on stale data -- treat signals with caution.",
                ticker, type(live_exc).__name__, age_min,
            )
            try:
                df = pd.read_json(entry["payload"]["json"], orient="split")
                df.columns = [str(c).lower() for c in df.columns]
                return df
            except Exception as e:
                logger.error("OHLCV cache fallback unreadable for {}: {}", ticker, e)
        logger.error("OHLCV unavailable for {} and no usable cache -- giving up.", ticker)
        raise


@with_retry(max_attempts=3, base_delay=3.0)
def get_info(ticker: str) -> dict:
    """Return company fundamentals dict from yfinance."""
    return _run_with_wall_timeout(
        lambda: yf.Ticker(ticker).info,
        _YF_WALL_TIMEOUT,
        ticker,
    )


# --- Alpaca helpers ----------------------------------------------------------

def get_alpaca_bars(
    ticker: str,
    days: int = 30,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Fetch recent bars from Alpaca's data API."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = _alpaca_data.get_stock_bars(req)
    df = bars.df
    if hasattr(df.index, "levels"):   # MultiIndex -> flatten
        df = df.droplevel(0)
    return df


def get_latest_quote(ticker: str) -> dict:
    """Return the latest bid/ask quote for a ticker."""
    req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
    quote = _alpaca_data.get_stock_latest_quote(req)[ticker]
    return {
        "ask_price": float(quote.ask_price),
        "bid_price": float(quote.bid_price),
        "ask_size":  int(quote.ask_size),
        "bid_size":  int(quote.bid_size),
        "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
    }
