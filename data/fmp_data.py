"""
data/fmp_data.py — Financial Modeling Prep (FMP) REST API wrapper.

Provides three capabilities needed by the event-based strategy engine:

  1. Earnings dates     — get_earnings_date(ticker)
     Used as the AVWAP anchor for Setup B (Institutional Breakout).
     Institutions reprice stocks after earnings; AVWAP from that date
     reflects the new price distribution the market has accepted.

  2. Float / shares     — get_float_shares(ticker)
     Used as the primary gate for Setup A (Small-Cap Expansion).
     Float < 50M is the defining characteristic of low-float runners.
     Float changes slowly; cached 7 days in memory.

  3. RS ranking universe — get_rs_universe()
     Returns ~600-1,400 tickers (S&P 500 + Nasdaq 100 + Russell 2000)
     used to rank each stock's relative strength against the broad market
     rather than only against the bot's custom watchlist.

Authentication:
    Set FMP_API_KEY in .env.  FMP Starter ($19/mo) covers all three endpoints.
    Free tier: 250 req/day (enough for development, not production scanning).

Rate limits:
    Starter: ~300 calls/min.  We use a 0.1s delay between batch calls and
    cache aggressively so the RS universe fetch (one call) and earnings date
    lookups (one call per ticker per day) stay well within budget.
"""
from __future__ import annotations

import os
import time
import threading
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Session + auth
# ---------------------------------------------------------------------------

_BASE = "https://financialmodelingprep.com/api/v3"
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


class FMPDisabledError(RuntimeError):
    """Raised when FMP_API_KEY is not configured."""


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        key = os.environ.get("FMP_API_KEY", "")
        if not key:
            try:
                from config import FMP_API_KEY
                key = FMP_API_KEY
            except ImportError:
                pass
        if not key:
            raise FMPDisabledError(
                "FMP_API_KEY is not set. Add it to .env to enable FMP data."
            )
        s = requests.Session()
        s.params = {"apikey": key}   # type: ignore[assignment]
        _SESSION = s
        logger.info("FMP session initialised (key: ...{})", key[-4:])
        return _SESSION


def _get(path: str, params: dict | None = None, timeout: int = 15) -> list | dict:
    url = f"{_BASE}{path}"
    resp = _get_session().get(url, params=params or {}, timeout=timeout)
    if resp.status_code == 401:
        raise PermissionError(f"FMP 401: invalid API key — {url}")
    if resp.status_code == 429:
        raise ConnectionError(f"FMP rate-limit (429) — {url}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# In-memory cache (TTL-based)
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str, ttl_sec: float) -> object | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < ttl_sec:
        return entry[1]
    return None


def _cache_set(key: str, value: object) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Earnings date  (AVWAP anchor for Setup B)
# ---------------------------------------------------------------------------

_EARNINGS_TTL_SEC = 24 * 3600   # refresh daily


def get_earnings_date(ticker: str) -> date | None:
    """
    Return the most recent earnings release date for `ticker`.
    Used as the anchor for earnings-anchored AVWAP in Setup B.

    Returns None when FMP is disabled, the ticker has no earnings history,
    or any network error occurs.

    Example:
        anchor = get_earnings_date("PLTR")
        # -> date(2026, 5, 5)
    """
    key = f"earn:{ticker.upper()}"
    cached = _cache_get(key, _EARNINGS_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        raw = _get(f"/historical/earning_calendar/{ticker.upper()}")
        # raw is a list of {"date": "2026-05-05", "symbol": ..., "eps": ...}
        if not isinstance(raw, list) or not raw:
            logger.debug("FMP | {} no earnings history", ticker)
            return None
        # Sort descending and take the most recent past date
        today = date.today()
        past = [
            r for r in raw
            if r.get("date") and r["date"][:10] <= today.isoformat()
        ]
        if not past:
            return None
        past.sort(key=lambda r: r["date"], reverse=True)
        d = date.fromisoformat(past[0]["date"][:10])
        _cache_set(key, d)
        logger.debug("FMP | {} earnings anchor: {}", ticker, d)
        return d
    except FMPDisabledError:
        raise
    except Exception as exc:
        logger.warning("FMP | get_earnings_date({}) failed: {}", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Float shares  (Setup A gate: float < 50M)
# ---------------------------------------------------------------------------

_FLOAT_TTL_SEC = 7 * 24 * 3600   # float changes slowly — refresh weekly


def get_float_shares(ticker: str) -> float | None:
    """
    Return the public float (number of shares freely tradeable) for `ticker`.
    Cached 7 days because float changes only on secondary offerings / buy-backs.

    Returns None when FMP is disabled or data is unavailable.

    Example:
        f = get_float_shares("KULR")
        # -> 38_200_000  (38.2M shares — qualifies for Setup A < 50M gate)
    """
    key = f"float:{ticker.upper()}"
    cached = _cache_get(key, _FLOAT_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        raw = _get(f"/shares_float", params={"symbol": ticker.upper()})
        # raw is a list with one dict: {"symbol":..., "floatShares":..., ...}
        if not isinstance(raw, list) or not raw:
            return None
        item = raw[0]
        val = item.get("floatShares") or item.get("freeFloat")
        if val is None:
            return None
        result = float(val)
        _cache_set(key, result)
        logger.debug("FMP | {} float: {:,.0f} shares", ticker, result)
        return result
    except FMPDisabledError:
        raise
    except Exception as exc:
        logger.warning("FMP | get_float_shares({}) failed: {}", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# RS ranking universe  (S&P 500 + Nasdaq 100 for Phase 1)
# ---------------------------------------------------------------------------
# Full Russell 2000 integration is Phase 2.  For Phase 1 we use the ~600
# unique tickers from SP500 + QQQ which FMP serves via two fast calls.

_UNIVERSE_TTL_SEC = 24 * 3600   # refresh daily at most


def get_rs_universe() -> list[str]:
    """
    Return the combined RS ranking universe: S&P 500 + Nasdaq 100 tickers.
    Deduplicated, uppercased.  Cached 24h.

    Phase 1: ~600 stocks (meaningful rank without the full Russell 2000 cost).
    Phase 2: extend to include Russell 2000 (~2,400 stocks total).

    Returns [] on failure — callers must handle an empty universe gracefully.
    """
    cached = _cache_get("rs_universe", _UNIVERSE_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore[return-value]

    tickers: set[str] = set()

    for endpoint in ("/sp500_constituent", "/nasdaq_constituent"):
        try:
            raw = _get(endpoint)
            if isinstance(raw, list):
                for item in raw:
                    sym = (item.get("symbol") or item.get("ticker") or "").upper().strip()
                    if sym:
                        tickers.add(sym)
            time.sleep(0.15)   # polite rate-limit gap
        except FMPDisabledError:
            raise
        except Exception as exc:
            logger.warning("FMP | get_rs_universe endpoint {} failed: {}", endpoint, exc)

    result = sorted(tickers)
    if result:
        _cache_set("rs_universe", result)
        logger.info("FMP | RS universe loaded: {} tickers (SP500 + Nasdaq100)", len(result))
    return result


# ---------------------------------------------------------------------------
# Relative Strength rank across the universe
# ---------------------------------------------------------------------------
# Computed from Tiingo OHLCV (already wired) rather than FMP to avoid burning
# FMP request quota on bulk OHLCV.  FMP is used only for the universe list.

_RS_RANKS_TTL_SEC = 4 * 3600   # recompute every 4 hours during market hours


def compute_rs_ranks(universe: list[str] | None = None) -> dict[str, float]:
    """
    Compute 20-day relative strength rank (0-100) for each ticker in `universe`.
    A rank of 90 means the stock outperformed 90% of the universe over 20 days.

    Returns {ticker: rank_0_to_100}.  Tickers with missing data are omitted.
    Cached 4 hours — expensive call (one OHLCV fetch per ticker).
    """
    cached = _cache_get("rs_ranks", _RS_RANKS_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore[return-value]

    if universe is None:
        universe = get_rs_universe()
    if not universe:
        return {}

    from data.tiingo_data import get_ohlcv, TiingoDisabledError

    returns: dict[str, float] = {}
    for i, ticker in enumerate(universe):
        try:
            df = get_ohlcv(ticker, period="2mo", interval="1d")
            if df is None or len(df) < 22:
                continue
            close = df["close"].squeeze()
            ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
            returns[ticker] = ret_20d
        except TiingoDisabledError:
            break
        except Exception:
            continue
        if i > 0 and i % 50 == 0:
            logger.debug("FMP | RS rank progress: {}/{}", i, len(universe))
        time.sleep(0.05)   # 50ms between calls — stays within Tiingo budget

    if not returns:
        return {}

    # Convert raw returns to percentile ranks (0-100)
    sorted_vals = sorted(returns.values())
    n = len(sorted_vals)
    ranks: dict[str, float] = {}
    for ticker, ret in returns.items():
        rank = sorted_vals.index(ret) / n * 100.0
        ranks[ticker] = round(rank, 1)

    _cache_set("rs_ranks", ranks)
    logger.info(
        "FMP | RS ranks computed: {} tickers  "
        "top5: {}",
        len(ranks),
        sorted(ranks, key=lambda t: -ranks[t])[:5],
    )
    return ranks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG")
    print("\n=== FMP self-test ===\n")

    print("--- PLTR earnings date ---")
    d = get_earnings_date("PLTR")
    print(f"  {d}")

    print("\n--- KULR float ---")
    f = get_float_shares("KULR")
    print(f"  {f:,.0f} shares" if f else "  N/A")

    print("\n--- RS universe (first 10) ---")
    u = get_rs_universe()
    print(f"  Total: {len(u)} | Sample: {u[:10]}")

    print("\n=== done ===\n")
