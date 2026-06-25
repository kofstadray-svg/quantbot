"""
agents/institutional_breakout.py â€” Setup B: Institutional Breakout screener.

The core thesis: institutions reprice growth stocks after earnings.  The stock
consolidates above the earnings-anchored VWAP while institutions accumulate.
When it breaks out of a 50-day high on above-average volume with ADX rising,
the institutional bid is expanding â€” this is where NVDA, PLTR, CRWD, HIMS
typically begin major moves.

ENTRY CONDITIONS (all must be True):
    1. SPY regime â€” bull or strong_bull
    2. RS Rank    â€” top 20% of universe (rank >= 80)
    3. ADX(14)    â€” > 25 and rising (measured vs 5 bars ago)
    4. AVWAP      â€” price above the earnings-anchored VWAP
    5. Volume     â€” 20-day avg volume > 150% of 60-day avg volume
    6. Breakout   â€” today's close above the 50-day high

EXIT PROFILE (strategy-specific, handled by exit_manager via EXIT_PROFILES):
    Initial stop : 2 ATR below entry
    Take profit 1: +2 ATR (sell 50%)
    Take profit 2: +4 ATR (sell 25%)
    Runner 25%   : Chandelier Exit trail

AVWAP anchor: most recent earnings release date (from FMP).
Fallback anchor when FMP unavailable: 63-day low (proxy for quarterly reset).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import ta
from loguru import logger

from data.market_data import get_ohlcv as _get_ohlcv

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_ADX           = 20.0    # trend must be established (lowered from 25 to double signal count)
ADX_RISING_BARS   = 3       # kept for logging context but NOT required for entry
RS_RANK_MIN       = 70.0    # top 20% of universe
VOL_EXPANSION_PCT = 1.20    # 20-day avg vol / 60-day avg vol >= 150%
BREAKOUT_WINDOW   = 30      # close must be above N-day high

# AVWAP fallback anchor: days back when no earnings date available
AVWAP_FALLBACK_DAYS = 63    # ~1 quarter


# ---------------------------------------------------------------------------
# AVWAP calculation
# ---------------------------------------------------------------------------

def _calc_avwap(df: pd.DataFrame, anchor_date: date) -> float | None:
    """
    Earnings-anchored VWAP.
    Anchor = first bar on or after anchor_date.
    Returns None if anchor is outside the available data range.
    """
    try:
        idx_dates = pd.to_datetime(df.index).date
        mask = [d >= anchor_date for d in idx_dates]
        sliced = df[mask]
        if len(sliced) < 2:
            return None
        typical = (sliced["high"] + sliced["low"] + sliced["close"]) / 3
        vol     = sliced["volume"]
        avwap   = float((typical * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
        return avwap
    except Exception as exc:
        logger.debug("IB | AVWAP calc failed: {}", exc)
        return None


def _get_anchor_date(ticker: str) -> date:
    """Return the earnings date from FMP, or the fallback 63-day-ago proxy."""
    try:
        from data.fmp_data import get_earnings_date, FMPDisabledError
        d = get_earnings_date(ticker)
        if d:
            return d
    except Exception:
        pass
    return date.today() - timedelta(days=AVWAP_FALLBACK_DAYS)


# ---------------------------------------------------------------------------
# Per-ticker signal
# ---------------------------------------------------------------------------

@dataclass
class IBSignal:
    ticker:          str
    strategy:        str = "institutional_breakout"
    action:          str = "BUY"
    score:           float = 0.0
    close:           float = 0.0
    entry:           float = 0.0          # breakout = yesterday's high
    stop_atr:        float = 0.0          # 2 ATR below entry
    rs_rank:         float = 0.0
    adx:             float = 0.0
    rvol:            float = 0.0          # 20d avg vol / 60d avg vol
    avwap:           float = 0.0
    avwap_anchor:    str   = ""
    high_50d:        float = 0.0
    atr:             float = 0.0
    fail_reasons:    list[str] = field(default_factory=list)


def _compute(
    ticker: str,
    rs_ranks: dict[str, float],
) -> IBSignal | None:
    """
    Compute all Setup B conditions for `ticker`.
    Returns IBSignal (pass or fail) or None on data error.
    rs_ranks: pre-computed {ticker: rank_0_to_100} from signal_router.
    """
    sig = IBSignal(ticker=ticker)
    fails: list[str] = []

    try:
        df = _get_ohlcv(ticker, period="1y", interval="1d")
        if df is None or len(df) < 65:
            return None
        df.columns = [c.lower() for c in df.columns]

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        last_close = float(close.iloc[-1])
        sig.close  = last_close

        # â”€â”€ 1. RS Rank â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rs_rank = rs_ranks.get(ticker.upper(), -1.0)
        sig.rs_rank = rs_rank
        if rs_rank < RS_RANK_MIN:
            fails.append(f"rs_rank={rs_rank:.0f} < {RS_RANK_MIN:.0f}")

        # â”€â”€ 2. ADX > 20 (no rising requirement) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            adx_series = ta.trend.ADXIndicator(
                high, low, close, window=14
            ).adx().dropna()
            adx_now  = float(adx_series.iloc[-1])
            adx_prev = float(adx_series.iloc[-1 - ADX_RISING_BARS])
            adx_rising = adx_now > adx_prev
            sig.adx = round(adx_now, 1)
        except Exception:
            adx_now, adx_rising = 0.0, False
            sig.adx = 0.0

        if adx_now <= MIN_ADX:
            fails.append(f"adx={adx_now:.1f} <= {MIN_ADX}")

        # â”€â”€ 3. Earnings-anchored VWAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        anchor_date = _get_anchor_date(ticker)
        avwap = _calc_avwap(df, anchor_date)
        sig.avwap        = round(avwap, 4) if avwap else 0.0
        sig.avwap_anchor = anchor_date.isoformat()

        if avwap is None:
            fails.append("avwap_unavailable")
        elif last_close <= avwap:
            fails.append(
                f"close={last_close:.2f} <= avwap={avwap:.2f} "
                f"(anchor={anchor_date})"
            )

        # â”€â”€ 4. Volume expansion: 20d avg > 150% of 60d avg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        vol_20 = float(volume.iloc[-20:].mean())
        vol_60 = float(volume.iloc[-60:].mean())
        rvol   = (vol_20 / vol_60) if vol_60 > 0 else 1.0
        sig.rvol = round(rvol, 2)

        if rvol < VOL_EXPANSION_PCT:
            fails.append(
                f"vol_expansion={rvol:.2f}x < {VOL_EXPANSION_PCT:.2f}x"
            )

        # â”€â”€ 5. 50-day high breakout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        high_50 = float(high.iloc[-BREAKOUT_WINDOW:-1].max())   # exclude today
        sig.high_50d = round(high_50, 4)

        if last_close <= high_50:
            fails.append(
                f"close={last_close:.2f} <= 50d_high={high_50:.2f}"
            )

        # â”€â”€ ATR for exit sizing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            atr_series = ta.volatility.AverageTrueRange(
                high, low, close, window=14
            ).average_true_range().dropna()
            atr = float(atr_series.iloc[-1])
        except Exception:
            atr = last_close * 0.02
        sig.atr      = round(atr, 4)
        sig.entry    = round(last_close, 4)          # trigger = close above 50d high
        sig.stop_atr = round(last_close - 2 * atr, 4)   # initial 2-ATR stop

        # â”€â”€ Score: count conditions passed (0-5) normalised to 0-100 â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conditions_passed = 5 - len([
            f for f in fails if not f.startswith("rs_rank") or rs_rank >= 0
        ])
        # RS rank contributes a continuous bonus on top of the binary check
        rs_bonus = max(0.0, (rs_rank - RS_RANK_MIN) / (100 - RS_RANK_MIN) * 20)
        sig.score = round(min(100.0, conditions_passed * 16 + rs_bonus), 1)
        sig.fail_reasons = fails

        return sig

    except Exception as exc:
        logger.warning("IB | {} compute failed: {}", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen(
    watchlist: list[str],
    rs_ranks:  dict[str, float] | None = None,
    regime=None,        # optional RegimeState from market_regime
) -> list[IBSignal]:
    """
    Screen `watchlist` for Setup B (Institutional Breakout) signals.

    Args:
        watchlist : ticker symbols to evaluate
        rs_ranks  : pre-computed RS ranks {ticker: 0-100}.
                    If None, computed fresh from FMP universe (expensive).
        regime    : optional RegimeState; blocks signals in bear/strong_bear.

    Returns:
        List of IBSignal where ALL 5 conditions pass, sorted by score desc.
        Failed signals are not returned (logged at DEBUG level only).
    """
    # â”€â”€ Regime gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trend = getattr(regime, "trend_regime", "neutral")
    if trend in ("bear", "strong_bear"):
        logger.info(
            "IB | REGIME VETO ({}): no institutional breakout signals in bear market",
            trend,
        )
        return []

    # â”€â”€ RS ranks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if rs_ranks is None:
        try:
            from data.fmp_data import compute_rs_ranks
            rs_ranks = compute_rs_ranks()
        except Exception as exc:
            logger.warning("IB | RS rank unavailable ({}), screening without rank gate", exc)
            rs_ranks = {}

    # â”€â”€ Per-ticker evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    passed: list[IBSignal] = []
    skipped = 0

    for ticker in watchlist:
        sig = _compute(ticker, rs_ranks)
        if sig is None:
            skipped += 1
            continue

        if sig.fail_reasons:
            logger.debug(
                "IB | {:<6} SKIP  score={:.0f}  fails=[{}]",
                ticker, sig.score, ", ".join(sig.fail_reasons),
            )
        else:
            logger.info(
                "IB | {:<6} [SIGNAL]  score={:.0f}  "
                "rs={:.0f}  adx={:.1f}  rvol={:.2f}x  "
                "close={:.2f} > avwap={:.2f}  "
                "close={:.2f} > 50dH={:.2f}  "
                "entry={:.2f}  stop={:.2f}  ATR={:.2f}",
                ticker, sig.score,
                sig.rs_rank, sig.adx, sig.rvol,
                sig.close, sig.avwap,
                sig.close, sig.high_50d,
                sig.entry, sig.stop_atr, sig.atr,
            )
            passed.append(sig)

    logger.info(
        "IB | scan complete: {}/{} passed  {} skipped  regime={}",
        len(passed), len(watchlist), skipped, trend,
    )
    passed.sort(key=lambda s: -s.score)
    return passed


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG")

    from data.custom_watchlist import CUSTOM_WATCHLIST
    try:
        from data.universe import get_nasdaq_watchlist, get_dow_watchlist
        watchlist = list(dict.fromkeys(CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()))
    except Exception:
        watchlist = CUSTOM_WATCHLIST

    print(f"\nInstitutional Breakout screen â€” {len(watchlist)} tickers\n")
    results = screen(watchlist)

    if not results:
        print("No Setup B signals today.")
    else:
        hdr = f"{'TICKER':<8} {'SCORE':>5}  {'RS':>5}  {'ADX':>5}  {'RVOL':>6}  {'AVWAP':>8}  {'ENTRY':>8}  {'STOP':>8}  {'ATR':>6}"
        print("â”€â”€ INSTITUTIONAL BREAKOUT SIGNALS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(hdr)
        print("â”€" * 80)
        for r in results:
            print(
                f"{r.ticker:<8} {r.score:>5.0f}  {r.rs_rank:>5.0f}  "
                f"{r.adx:>5.1f}  {r.rvol:>6.2f}x  "
                f"{r.avwap:>8.2f}  {r.entry:>8.2f}  "
                f"{r.stop_atr:>8.2f}  {r.atr:>6.2f}"
            )
    print(f"\nTotal: {len(results)} signals")
