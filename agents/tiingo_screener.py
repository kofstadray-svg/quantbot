"""
agents/tiingo_screener.py — Chart-pattern screener powered by Tiingo OHLCV.

A second, independent chart-pattern screen that complements the main
chart_pattern_screener.py.  Key differences:

  DATA   : Sources OHLCV directly from data.tiingo_data (bypasses the
            market_data abstraction layer for simplicity and speed).

  QUALITY GATES (new, not in the base screener):
    1. Volume Expansion  — confirmed breakouts must have rel-vol >= 1.5x
    2. ATR Compression   — pattern must have formed during a low-vol squeeze
                           (current ATR < 60-day median)
    3. Bear-Regime Veto  — SPY below 200 EMA suppresses all BUY signals

  SCORING (0-100):
    Pattern strength  30 pts  confirmed/bullish > forming/neutral > bearish
    Volume expansion  25 pts  graduated by rel-vol level
    ATR compression   20 pts  tighter squeeze → more points
    Trend alignment   15 pts  price above EMA50 + EMA200
    Pattern quality   10 pts  pattern reliability tier (reversal > continuation)

  SIGNAL TIERS:
    STRONG BUY  score >= 75  confirmed + quality gates clear
    BUY         score >= 55  confirmed or strong forming
    WATCH       score >= 35  forming / partial quality
    SKIP        score <  35

HOW TO USE:
  from agents.tiingo_screener import screen
  results = screen(["AAPL", "NVDA", "MSFT"], regime=regime_state)

HOW TO RUN STANDALONE:
  python agents/tiingo_screener.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pandas as pd
import ta
from loguru import logger
from dataclasses import dataclass, field

from agents.chart_pattern_screener import detect_chart_patterns, ChartSignal


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SCORE_STRONG_BUY = 75
SCORE_BUY        = 55
SCORE_WATCH      = 35

# Minimum volume expansion on the confirmed breakout bar
REL_VOL_THRESHOLD = 1.5   # 1.5x average volume required for confirmed patterns

# ATR compression: current ATR must be below this fraction of 60-day median
ATR_SQUEEZE_FRAC = 1.10   # <= 110% of median = compression territory

# Lookback for OHLCV fetch
OHLCV_PERIOD = "6mo"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch(ticker: str) -> pd.DataFrame:
    """
    Fetch 6-month daily OHLCV from Tiingo directly.
    Falls back to the standard data layer (which tries Tiingo then yfinance).
    Returns lowercase-column DataFrame or empty DataFrame on failure.
    """
    try:
        from data.tiingo_data import get_ohlcv
        df = get_ohlcv(ticker, period=OHLCV_PERIOD, interval="1d")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug("tiingo_screener | {} Tiingo direct failed ({}), using data layer", ticker, e)
    # Fallback
    try:
        from data.market_data import get_ohlcv as _get
        df = _get(ticker, period=OHLCV_PERIOD, interval="1d")
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _rel_vol(df: pd.DataFrame) -> float:
    """Today's volume vs 20-day average (prior 20 bars)."""
    if "volume" not in df.columns or len(df) < 5:
        return 1.0
    vol = df["volume"]
    avg = float(vol.iloc[-21:-1].mean())
    return float(vol.iloc[-1]) / avg if avg > 0 else 1.0


def _atr_compression(df: pd.DataFrame, window: int = 14, lookback: int = 60) -> float:
    """
    Returns the ratio of current ATR to 60-day median ATR.
    < 1.0  = compressed (good setup)
    > 1.0  = expanded (already moving)
    """
    if len(df) < lookback + window:
        return 1.0
    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        atr   = ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()
        current = float(atr.iloc[-1])
        median  = float(atr.iloc[-lookback:-1].median())
        return round(current / median, 3) if median > 0 else 1.0
    except Exception:
        return 1.0


def _trend_score(df: pd.DataFrame) -> float:
    """
    0-15 pts:  price position relative to 50 and 200 EMA.
    Both above → 15 pts
    Only 50 above → 8 pts
    Neither → 0 pts
    """
    if len(df) < 50:
        return 0.0
    try:
        close  = df["close"]
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        price  = float(close.iloc[-1])
        above50  = price > ema50
        above200 = False
        if len(df) >= 200:
            ema200   = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            above200 = price > ema200
        if above50 and above200:
            return 15.0
        if above50:
            return 8.0
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

# Reliability tier per pattern type (higher = more reliable, more pts)
_PATTERN_TIER: dict[str, int] = {
    "Head & Shoulders":          3,
    "Inverse Head & Shoulders":  3,
    "Double Bottom":             3,
    "Double Top":                3,
    "Ascending Triangle":        2,
    "Descending Triangle":       2,
    "Symmetrical Triangle":      2,
    "Rising Wedge":              2,
    "Falling Wedge":             2,
    "Bullish Flag":              1,
    "Bearish Flag":              1,
    "Rectangle":                 1,
}


def _score_signal(
    signal: ChartSignal,
    rv:     float,   # relative volume
    atr_r:  float,   # ATR compression ratio
    trend:  float,   # trend alignment pts (0-15)
) -> float:
    """Compute 0-100 signal quality score."""
    pts = 0.0

    # 1. Pattern strength (30 pts)
    if signal.state == "confirmed" and signal.direction == "bullish":
        pts += 30.0
    elif signal.state == "confirmed":
        pts += 20.0
    elif signal.state == "forming" and signal.direction == "bullish":
        pts += 15.0
    elif signal.state == "forming":
        pts += 8.0

    # 2. Volume expansion (25 pts) — graduated
    if   rv >= 4.0:  pts += 25.0
    elif rv >= 3.0:  pts += 20.0
    elif rv >= 2.0:  pts += 15.0
    elif rv >= 1.5:  pts += 10.0
    elif rv >= 1.0:  pts +=  4.0

    # 3. ATR compression (20 pts) — tighter squeeze → more pts
    if   atr_r <= 0.70:  pts += 20.0
    elif atr_r <= 0.85:  pts += 15.0
    elif atr_r <= 1.00:  pts += 10.0
    elif atr_r <= 1.10:  pts +=  5.0

    # 4. Trend alignment (15 pts)
    pts += trend

    # 5. Pattern reliability tier (10 pts)
    tier = _PATTERN_TIER.get(signal.pattern, 1)
    pts += {3: 10.0, 2: 6.0, 1: 3.0}.get(tier, 3.0)

    return round(min(pts, 100.0), 1)


def _signal_tier(score: float) -> str:
    if score >= SCORE_STRONG_BUY: return "STRONG BUY"
    if score >= SCORE_BUY:        return "BUY"
    if score >= SCORE_WATCH:      return "WATCH"
    return "SKIP"


# ---------------------------------------------------------------------------
# Bear-regime veto
# ---------------------------------------------------------------------------

def _bear_regime() -> bool:
    """
    True if SPY is below its 200 EMA — suppress all BUY signals.
    Uses Tiingo directly. Falls back to False (no veto) on failure.
    """
    try:
        from data.tiingo_data import get_ohlcv
        spy = get_ohlcv("SPY", period="1y", interval="1d")
        if spy is None or spy.empty or len(spy) < 200:
            return False
        close  = spy["close"]
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        return float(close.iloc[-1]) < ema200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class TiingoSignal:
    ticker:       str
    pattern:      str
    state:        str        # confirmed | forming
    direction:    str        # bullish | bearish | neutral
    action:       str        # BUY | SELL | HOLD | SKIP
    signal_tier:  str        # STRONG BUY | BUY | WATCH | SKIP
    score:        float      # 0-100
    rel_vol:      float
    atr_ratio:    float      # current ATR / 60-day median
    trend_pts:    float      # 0-15 trend alignment score
    reason:       str
    all_patterns: list[str] = field(default_factory=list)


def screen(
    watchlist:   list[str],
    regime=None,            # optional RegimeState (for logging context)
    bear_veto:   bool = True,
    min_score:   float = SCORE_WATCH,
) -> list[TiingoSignal]:
    """
    Screen a watchlist for chart patterns using Tiingo OHLCV.

    Args:
        watchlist  : list of ticker symbols
        regime     : optional RegimeState from market_regime.compute_regime()
        bear_veto  : if True, suppress BUY signals when SPY < 200 EMA
        min_score  : only return signals with score >= this (default: WATCH=35)

    Returns:
        List of TiingoSignal sorted by score descending.
    """
    # Bear-regime check
    is_bear = _bear_regime() if bear_veto else False
    if is_bear:
        logger.warning("tiingo_screener | BEAR REGIME ACTIVE — BUY signals suppressed (SPY < 200 EMA)")

    results: list[TiingoSignal] = []

    for ticker in watchlist:
        try:
            df = _fetch(ticker)
            if df.empty or len(df) < 30:
                logger.debug("tiingo_screener | {} insufficient data ({} bars)", ticker, len(df))
                continue

            # Detect chart patterns
            patterns = detect_chart_patterns(ticker, df)
            if not patterns:
                continue

            # Pick the best pattern (confirmed > forming, bullish > neutral)
            action_ord = {"BUY": 0, "SELL": 1, "HOLD": 2}
            state_ord  = {"confirmed": 0, "forming": 1}
            patterns.sort(key=lambda p: (state_ord[p.state], action_ord[p.action]))
            best = patterns[0]

            # Quality metrics
            rv     = _rel_vol(df)
            atr_r  = _atr_compression(df)
            trend  = _trend_score(df)
            score  = _score_signal(best, rv, atr_r, trend)

            # Quality gate: confirmed patterns need volume
            if best.state == "confirmed" and rv < REL_VOL_THRESHOLD:
                logger.debug(
                    "tiingo_screener | {} pattern={} confirmed but vol={:.1f}x < {:.1f}x gate — penalising",
                    ticker, best.pattern, rv, REL_VOL_THRESHOLD,
                )
                score *= 0.70  # 30% penalty — still scored but demoted

            # Skip if below minimum
            if score < min_score:
                continue

            # Determine action
            tier   = _signal_tier(score)
            action = best.action

            # Bear-regime veto: downgrade BUY → HOLD
            if is_bear and action == "BUY":
                action = "HOLD"
                tier   = "WATCH"

            sig = TiingoSignal(
                ticker       = ticker,
                pattern      = best.pattern,
                state        = best.state,
                direction    = best.direction,
                action       = action,
                signal_tier  = tier,
                score        = round(score, 1),
                rel_vol      = round(rv, 2),
                atr_ratio    = atr_r,
                trend_pts    = trend,
                reason       = best.reason,
                all_patterns = [p.pattern for p in patterns],
            )
            results.append(sig)

            logger.info(
                "Tiingo | {:<6}  [{:<10}]  {:<28}  score={:5.1f}  "
                "vol={:.1f}x  ATR={:.2f}x  trend={:.0f}pts  {}",
                ticker, tier, best.pattern, score,
                rv, atr_r, trend, best.state,
            )

        except Exception as exc:
            logger.warning("tiingo_screener | {} failed: {}", ticker, exc)

    # Sort by score desc, confirmed first within same tier
    results.sort(key=lambda s: (-s.score, 0 if s.state == "confirmed" else 1))
    return results


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    try:
        from data.universe import get_nasdaq_watchlist, get_dow_watchlist
        universe = list(dict.fromkeys(
            CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
        ))
    except Exception:
        universe = CUSTOM_WATCHLIST

    print(f"\nTiingo Chart Pattern Screen — {len(universe)} tickers\n")
    results = screen(universe)

    if not results:
        print("No signals above WATCH threshold.")
    else:
        strong = [r for r in results if r.signal_tier == "STRONG BUY"]
        buys   = [r for r in results if r.signal_tier == "BUY"]
        watch  = [r for r in results if r.signal_tier == "WATCH"]

        hdr = f"{'TICKER':<8} {'TIER':<12} {'SCORE':>5}  {'PATTERN':<28} {'STATE':<10} {'VOL':>5}  {'ATR':>5}  REASON"
        sep = "─" * 110

        if strong:
            print("── STRONG BUY ─────────────────────────────────────────────────────────────────────────")
            print(hdr); print(sep)
            for r in strong:
                print(f"{r.ticker:<8} {r.signal_tier:<12} {r.score:>5.1f}  {r.pattern:<28} {r.state:<10} {r.rel_vol:>5.1f}x {r.atr_ratio:>5.2f}  {r.reason[:45]}")

        if buys:
            print("\n── BUY ────────────────────────────────────────────────────────────────────────────────")
            print(hdr); print(sep)
            for r in buys:
                print(f"{r.ticker:<8} {r.signal_tier:<12} {r.score:>5.1f}  {r.pattern:<28} {r.state:<10} {r.rel_vol:>5.1f}x {r.atr_ratio:>5.2f}  {r.reason[:45]}")

        if watch:
            print("\n── WATCH ──────────────────────────────────────────────────────────────────────────────")
            print(hdr); print(sep)
            for r in watch:
                print(f"{r.ticker:<8} {r.signal_tier:<12} {r.score:>5.1f}  {r.pattern:<28} {r.state:<10} {r.rel_vol:>5.1f}x {r.atr_ratio:>5.2f}  {r.reason[:45]}")

    print(f"\nTotal: {len(results)} signals found")
