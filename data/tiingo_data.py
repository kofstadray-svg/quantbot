"""
data/tiingo_data.py — Tiingo REST API wrapper.

Replaces yfinance as the primary OHLCV provider.
Drop-in interface: callers that previously used yf.download() should switch
to get_ohlcv() / get_crypto_ohlcv() / get_batch_ohlcv() from this module.

Tiingo REST API (v1)
====================
  Daily OHLCV:   GET /tiingo/daily/{ticker}/prices?startDate=&endDate=&token=
  Intraday:      GET /iex/{ticker}/prices?resampleFreq=5min&token=
  Crypto daily:  GET /tiingo/crypto/prices?tickers=btcusd&startDate=&endDate=&token=
  Metadata:      GET /tiingo/daily/{ticker}?token=

Rate limits (Free tier): 50 req/hr, 1 000 req/day
Rate limits (Power $10): 1 000 req/hr, 100 000 req/day   <- recommended
Rate limits (Starter $30): 5 000 req/hr, unlimited daily

Authentication: Authorization: Token KEY header (key never appears in logs/URLs)

HOW TO ENABLE
=============
Add to .env:
    TIINGO_API_KEY=your_key_here

When TIINGO_API_KEY is blank the module raises TiingoDisabledError on any
call so callers can catch it and fall back to yfinance.
"""
from __future__ import annotations

import os
import time
import threading
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Auth & session
# ---------------------------------------------------------------------------

_BASE = "https://api.tiingo.com"
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


class TiingoDisabledError(RuntimeError):
    """Raised when TIINGO_API_KEY is not set."""


def _get_session() -> requests.Session:
    """Return a cached requests.Session with the auth header pre-loaded."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        key = os.environ.get("TIINGO_API_KEY", "")
        if not key:
            try:
                from config import TIINGO_API_KEY
                key = TIINGO_API_KEY
            except ImportError:
                pass
        if not key:
            raise TiingoDisabledError(
                "TIINGO_API_KEY is not set. Add it to .env or set the "
                "environment variable before starting the bot."
            )
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Token {key}",
            "Content-Type":  "application/json",
        })
        _SESSION = s
        logger.info("Tiingo session initialised (key: ...{})", key[-4:])
        return _SESSION


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 20

# ---------------------------------------------------------------------------
# Circuit breaker — stop hammering Tiingo after repeated 429s
# ---------------------------------------------------------------------------
# After _CB_THRESHOLD consecutive 429s the breaker opens and all calls
# fall through immediately to yfinance for _CB_COOLDOWN_SEC seconds.
# This avoids 7,000+ "attempt 1/3 → 2/3 → 3/3" log lines when Tiingo's
# daily quota is exhausted (e.g. day after heavy backtest runs).
import time as _time_mod

_CB_THRESHOLD    = 5      # consecutive 429s before opening
_CB_COOLDOWN_SEC = 3600   # 1 hour cooldown
_cb_fail_count   = 0
_cb_open_until   = 0.0    # epoch seconds; 0 = closed (normal)


def _cb_record_429():
    global _cb_fail_count, _cb_open_until
    _cb_fail_count += 1
    if _cb_fail_count >= _CB_THRESHOLD:
        _cb_open_until = _time_mod.time() + _CB_COOLDOWN_SEC
        logger.warning(
            "Tiingo circuit-breaker OPEN — {} consecutive 429s, "
            "falling back to yfinance for next {}min",
            _cb_fail_count, _CB_COOLDOWN_SEC // 60,
        )
        _cb_fail_count = 0   # reset so it re-opens cleanly after cooldown


def _cb_record_success():
    global _cb_fail_count
    _cb_fail_count = 0


def _cb_is_open() -> bool:
    """Return True when the breaker is open (Tiingo should be skipped)."""
    global _cb_open_until
    if _cb_open_until and _time_mod.time() < _cb_open_until:
        return True
    if _cb_open_until and _time_mod.time() >= _cb_open_until:
        _cb_open_until = 0.0
        logger.info("Tiingo circuit-breaker CLOSED — resuming Tiingo requests")
    return False


def _get(path: str, params: dict | None = None, timeout: int = _DEFAULT_TIMEOUT):
    """GET {_BASE}{path} and return parsed JSON.
    Retries up to 3 times on 429 with exponential back-off (1s, 2s, 4s).
    Circuit breaker opens after _CB_THRESHOLD consecutive 429s to avoid
    flooding logs when the daily quota is exhausted.
    Raises on non-2xx after retries exhausted.
    """
    if _cb_is_open():
        raise ConnectionError("Tiingo circuit-breaker open — using yfinance fallback")

    url = f"{_BASE}{path}"
    for attempt in range(3):
        resp = _get_session().get(url, params=params or {}, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt          # 1 s, 2 s, 4 s
            _cb_record_429()
            logger.warning(
                "Tiingo rate-limit (429) for {} -- waiting {}s (attempt {}/3)",
                url, wait, attempt + 1,
            )
            _time_mod.sleep(wait)
            continue
        if resp.status_code == 404:
            raise ValueError(f"Tiingo 404: {url} — ticker may be invalid or delisted")
        resp.raise_for_status()
        _cb_record_success()
        return resp.json()
    _cb_record_429()
    raise ConnectionError(f"Tiingo rate-limit (429): {url} — exhausted retries")


# ---------------------------------------------------------------------------
# In-memory session cache
# ---------------------------------------------------------------------------
# Prevents re-fetching the same ticker+period within the same process.
# The regime compute runs every 30 min and fetches SPY/QQQ/RSP/XLK/XLU.
# Without this, burst backtest runs and live scans both hammer Tiingo and
# trigger 429 rate-limit responses. TTL matches the regime cache (30 min).
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


def _get(path: str, params: dict | None = None, timeout: int = _DEFAULT_TIMEOUT):
    """GET {_BASE}{path} and return parsed JSON.
    Retries up to 3 times on 429 with exponential back-off (1s, 2s, 4s).
    Raises on non-2xx after retries exhausted.
    """
    url = f"{_BASE}{path}"
    for attempt in range(3):
        resp = _get_session().get(url, params=params or {}, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt          # 1 s, 2 s, 4 s
            logger.warning(
                "Tiingo rate-limit (429) for {} -- waiting {}s (attempt {}/3)",
                url, wait, attempt + 1,
            )
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            raise ValueError(f"Tiingo 404: {url} — ticker may be invalid or delisted")
        resp.raise_for_status()
        return resp.json()
    raise ConnectionError(f"Tiingo rate-limit (429): {url} — exhausted retries")


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _period_to_start(period: str) -> date:
    """Convert yfinance-style period strings to a start date."""
    today = date.today()
    p = period.lower().strip()
    if p.endswith("d"):
        return today - timedelta(days=int(p[:-1]))
    if p.endswith("wk"):
        return today - timedelta(weeks=int(p[:-2]))
    if p.endswith("mo"):
        return today - timedelta(days=int(p[:-2]) * 31)
    if p.endswith("m") and not p.endswith("mo"):
        return today - timedelta(days=int(p[:-1]) * 31)
    if p.endswith("y"):
        return today - timedelta(days=int(p[:-1]) * 366)
    raise ValueError(f"Unrecognised period string: {period!r}")


# ---------------------------------------------------------------------------
# Daily OHLCV — equities
# ---------------------------------------------------------------------------

def get_ohlcv(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Fetch daily (or intraday) OHLCV for an equity ticker.
    Returns DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume, adj_close
    Empty DataFrame on total failure.
    """
    if interval != "1d":
        return _get_intraday(ticker, interval=interval)

    end_dt   = end   or date.today()
    start_dt = start or _period_to_start(period)

    # Check in-memory session cache first (avoids 429 during burst scans)
    mem_key = f"eq:{ticker.upper()}:{period}:{interval}:{end_dt}"
    cached = _mem_get(mem_key)
    if cached is not None:
        logger.debug("Tiingo | {} daily [mem-cache hit]", ticker)
        return cached

    logger.info("Tiingo | {} daily  {} -> {}", ticker, start_dt, end_dt)
    raw = _get(
        f"/tiingo/daily/{ticker.upper()}/prices",
        params={
            "startDate":    start_dt.isoformat(),
            "endDate":      end_dt.isoformat(),
            "resampleFreq": "daily",
        },
    )

    if not raw:
        logger.warning("Tiingo | {} returned empty daily data", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open":       "open",
        "high":       "high",
        "low":        "low",
        "close":      "close",
        "volume":     "volume",
        "adjClose":   "adj_close",
        "adjOpen":    "adj_open",
        "adjHigh":    "adj_high",
        "adjLow":     "adj_low",
        "adjVolume":  "adj_volume",
    })
    keep = [c for c in ("open", "high", "low", "close", "volume", "adj_close") if c in df.columns]
    result = df[keep].dropna(subset=["close"])
    _mem_put(mem_key, result)
    return result


# ---------------------------------------------------------------------------
# Intraday — IEX endpoint
# ---------------------------------------------------------------------------

def _get_intraday(ticker: str, interval: str = "5min") -> pd.DataFrame:
    """Fetch intraday bars from Tiingo IEX (last ~8 trading hours)."""
    logger.info("Tiingo | {} intraday {}", ticker, interval)
    raw = _get(
        f"/iex/{ticker.upper()}/prices",
        params={"resampleFreq": interval, "columns": "date,open,high,low,close,volume"},
    )
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# Batch OHLCV
# ---------------------------------------------------------------------------

def get_batch_ohlcv(
    tickers: list[str],
    period:  str = "3mo",
    interval: str = "1d",
    *,
    delay_between: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for multiple tickers sequentially.
    Returns {ticker: DataFrame}; failed tickers map to empty DataFrames.
    delay_between: seconds between requests (rate-limit guard).
    """
    result: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        try:
            result[ticker] = get_ohlcv(ticker, period=period, interval=interval)
        except TiingoDisabledError:
            raise
        except Exception as exc:
            logger.warning("Tiingo | batch failed for {}: {}", ticker, exc)
            result[ticker] = pd.DataFrame()
        if i < len(tickers) - 1 and delay_between > 0:
            time.sleep(delay_between)
    return result


# ---------------------------------------------------------------------------
# Crypto OHLCV
# ---------------------------------------------------------------------------

def _yf_to_tiingo_crypto(symbol: str) -> str:
    """'BTC-USD' -> 'btcusd',  'SHIB-USD' -> 'shibusd'"""
    return symbol.lower().replace("-", "")


def get_crypto_ohlcv(
    symbol: str,
    period:   str = "6mo",
    interval: str = "1d",
    *,
    start: date | None = None,
    end:   date | None = None,
) -> pd.DataFrame:
    """
    Fetch daily crypto OHLCV from Tiingo.
    symbol: yfinance-style 'BTC-USD', 'ETH-USD', etc.
    Returns DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    tiingo_sym = _yf_to_tiingo_crypto(symbol)
    end_dt   = end   or date.today()
    start_dt = start or _period_to_start(period)

    mem_key = f"cr:{tiingo_sym}:{period}:{interval}:{end_dt}"
    cached = _mem_get(mem_key)
    if cached is not None:
        logger.debug("Tiingo | crypto {} [mem-cache hit]", symbol)
        return cached

    logger.info("Tiingo | crypto {} ({}) {} -> {}", symbol, tiingo_sym, start_dt, end_dt)
    raw = _get(
        "/tiingo/crypto/prices",
        params={
            "tickers":      tiingo_sym,
            "startDate":    start_dt.isoformat(),
            "endDate":      end_dt.isoformat(),
            "resampleFreq": "1day",
        },
    )

    if not raw:
        logger.warning("Tiingo | crypto {} empty response", symbol)
        return pd.DataFrame()

    price_data = raw[0].get("priceData", []) if isinstance(raw, list) else []
    if not price_data:
        logger.warning("Tiingo | crypto {} empty priceData", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(price_data)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"volumeNotional": "volume_notional", "tradesDone": "trades"})
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    result = df[keep].dropna(subset=["close"])
    _mem_put(mem_key, result)
    return result


# ---------------------------------------------------------------------------
# Metadata & latest price
# ---------------------------------------------------------------------------

def get_ticker_meta(ticker: str) -> dict:
    """Return Tiingo metadata: name, exchange, startDate, endDate."""
    try:
        raw = _get(f"/tiingo/daily/{ticker.upper()}")
        return {
            "name":       raw.get("name", ""),
            "exchange":   raw.get("exchangeCode", ""),
            "start_date": raw.get("startDate", ""),
            "end_date":   raw.get("endDate", ""),
            "ticker":     raw.get("ticker", ticker),
        }
    except Exception as exc:
        logger.warning("Tiingo | get_ticker_meta({}) failed: {}", ticker, exc)
        return {}


def get_latest_price(ticker: str) -> float | None:
    """Return the most recent adjusted-close price. None on failure."""
    try:
        raw = _get(
            f"/tiingo/daily/{ticker.upper()}/prices",
            params={"startDate": (date.today() - timedelta(days=5)).isoformat()},
        )
        if raw:
            return float(raw[-1].get("adjClose") or raw[-1].get("close") or 0)
    except Exception as exc:
        logger.warning("Tiingo | get_latest_price({}) failed: {}", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Self-test (python data/tiingo_data.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG")
    print("\n=== Tiingo self-test ===\n")

    print("--- AAPL daily 1mo ---")
    df = get_ohlcv("AAPL", period="1mo")
    if df.empty:
        print("  FAILED -- empty DataFrame")
    else:
        print(f"  {len(df)} rows  |  {df.index[0].date()} -> {df.index[-1].date()}")
        print(f"  latest close: ${df['close'].iloc[-1]:.2f}")

    print("\n--- SPY daily 3mo ---")
    df2 = get_ohlcv("SPY", period="3mo")
    print(f"  {len(df2)} rows  |  latest close: ${df2['close'].iloc[-1]:.2f}" if not df2.empty else "  FAILED")

    print("\n--- BTC-USD crypto 1mo ---")
    df3 = get_crypto_ohlcv("BTC-USD", period="1mo")
    if df3.empty:
        print("  FAILED -- empty DataFrame")
    else:
        print(f"  {len(df3)} rows  |  latest close: ${df3['close'].iloc[-1]:,.2f}")

    print("\n--- batch MSFT + GOOGL 1mo ---")
    batch = get_batch_ohlcv(["MSFT", "GOOGL"], period="1mo")
    for sym, d in batch.items():
        status = f"{len(d)} rows, latest ${d['close'].iloc[-1]:.2f}" if not d.empty else "FAILED"
        print(f"  {sym}: {status}")

    print("\n--- NVDA metadata ---")
    print(f"  {get_ticker_meta('NVDA')}")
    print("\n=== done ===\n")
