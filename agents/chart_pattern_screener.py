"""
agents/chart_pattern_screener.py — Chart pattern detector.

Detects chart patterns from "The Only Technical Analysis Book You Will Ever Need"
(Brian Hale) using 90-day OHLCV data.

Patterns detected:
  REVERSAL:     Double Top, Double Bottom, Head & Shoulders, Inverse H&S,
                Rising Wedge, Falling Wedge
  CONTINUATION: Ascending Triangle, Descending Triangle, Symmetrical Triangle,
                Bullish Flag, Bearish Flag, Rectangle

Signal states:
  "confirmed"  → price has already broken through the key level (trade now)
  "forming"    → pattern is developing, not yet confirmed (watch list)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

# ── scipy optional — fall back to a pure-numpy peak finder ────────────────────
try:
    from scipy.signal import find_peaks as _sp_peaks
except ImportError:
    def _sp_peaks(arr: np.ndarray, distance: int = 1):  # type: ignore[misc]
        """Minimal drop-in: returns indices of local maxima separated by `distance` bars."""
        arr = np.asarray(arr, dtype=float)
        n = len(arr)
        peaks = []
        i = 1
        while i < n - 1:
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(i)
                i += distance  # skip ahead to enforce minimum distance
            else:
                i += 1
        return np.array(peaks, dtype=int), {}

# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class ChartSignal:
    ticker:    str
    pattern:   str
    direction: str      # "bullish" | "bearish" | "neutral"
    state:     str      # "confirmed" | "forming"
    action:    str      # "BUY" | "SELL" | "HOLD"
    reason:    str


# ── Peak / trough detection ────────────────────────────────────────────────────

def _peaks(arr: np.ndarray, min_dist: int = 5) -> np.ndarray:
    """Return indices of local maxima, separated by at least min_dist bars."""
    idx, _ = _sp_peaks(arr, distance=min_dist)
    return idx

def _troughs(arr: np.ndarray, min_dist: int = 5) -> np.ndarray:
    """Return indices of local minima, separated by at least min_dist bars."""
    idx, _ = _sp_peaks(-arr, distance=min_dist)
    return idx

def _trendline(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit OLS line to (x, y) pairs; return (slope, intercept)."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    coeffs = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(coeffs[0]), float(coeffs[1])

def _pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference between two values."""
    if b == 0:
        return 0.0
    return abs(a - b) / b


# ── Pattern detectors ──────────────────────────────────────────────────────────

def _double_top(close: np.ndarray, tol: float = 0.03) -> tuple[bool, bool]:
    """
    Detect Double Top (M-shape): two peaks near the same price after an uptrend.
    Returns (forming, confirmed).
    Confirmed when price has closed below the neckline (trough between peaks).
    """
    pk = _peaks(close, min_dist=5)
    if len(pk) < 2:
        return False, False
    p1, p2 = int(pk[-2]), int(pk[-1])
    if _pct_diff(close[p1], close[p2]) > tol:
        return False, False
    # Neckline = minimum between the two peaks
    neckline = float(close[p1:p2 + 1].min())
    current  = float(close[-1])
    forming   = current > neckline
    confirmed = current < neckline
    return forming, confirmed

def _double_bottom(close: np.ndarray, tol: float = 0.03) -> tuple[bool, bool]:
    """Detect Double Bottom (W-shape): two troughs near the same price."""
    tr = _troughs(close, min_dist=5)
    if len(tr) < 2:
        return False, False
    t1, t2 = int(tr[-2]), int(tr[-1])
    if _pct_diff(close[t1], close[t2]) > tol:
        return False, False
    neckline = float(close[t1:t2 + 1].max())
    current  = float(close[-1])
    forming   = current < neckline
    confirmed = current > neckline
    return forming, confirmed

def _head_and_shoulders(close: np.ndarray, tol: float = 0.04) -> tuple[bool, bool]:
    """
    Detect Head & Shoulders: three peaks, middle (head) is highest,
    left and right shoulders at approximately the same level.
    """
    pk = _peaks(close, min_dist=5)
    if len(pk) < 3:
        return False, False
    ls, hd, rs = int(pk[-3]), int(pk[-2]), int(pk[-1])
    # Head must be the tallest
    if not (close[hd] > close[ls] and close[hd] > close[rs]):
        return False, False
    # Shoulders roughly equal
    if _pct_diff(close[ls], close[rs]) > tol:
        return False, False
    # Neckline = average of troughs between shoulders
    trough1 = float(close[ls:hd + 1].min())
    trough2 = float(close[hd:rs + 1].min())
    neckline = (trough1 + trough2) / 2
    current  = float(close[-1])
    forming   = current > neckline
    confirmed = current < neckline
    return forming, confirmed

def _inverse_head_and_shoulders(close: np.ndarray, tol: float = 0.04) -> tuple[bool, bool]:
    """Detect Inverse Head & Shoulders: three troughs, middle is lowest."""
    tr = _troughs(close, min_dist=5)
    if len(tr) < 3:
        return False, False
    ls, hd, rs = int(tr[-3]), int(tr[-2]), int(tr[-1])
    if not (close[hd] < close[ls] and close[hd] < close[rs]):
        return False, False
    if _pct_diff(close[ls], close[rs]) > tol:
        return False, False
    peak1    = float(close[ls:hd + 1].max())
    peak2    = float(close[hd:rs + 1].max())
    neckline = (peak1 + peak2) / 2
    current  = float(close[-1])
    forming   = current < neckline
    confirmed = current > neckline
    return forming, confirmed

def _ascending_triangle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int = 30
) -> tuple[bool, bool]:
    """
    Ascending triangle: flat resistance (peaks at same level) +
    rising support (higher lows).
    """
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False

    pk_vals = high[-n:][pk]
    tr_vals = low[-n:][tr]

    # Resistance is flat (peaks within 2% of each other)
    if _pct_diff(float(pk_vals.max()), float(pk_vals.min())) > 0.02:
        return False, False

    # Support is rising (positive slope through troughs)
    tr_slope, _ = _trendline(tr.astype(float), tr_vals)
    if tr_slope <= 0:
        return False, False

    resistance = float(pk_vals.mean())
    current    = float(close[-1])
    confirmed  = current > resistance
    forming    = not confirmed and current > resistance * 0.97
    return forming, confirmed

def _descending_triangle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int = 30
) -> tuple[bool, bool]:
    """
    Descending triangle: flat support (troughs at same level) +
    falling resistance (lower highs).
    """
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False

    pk_vals = high[-n:][pk]
    tr_vals = low[-n:][tr]

    # Support is flat
    if _pct_diff(float(tr_vals.max()), float(tr_vals.min())) > 0.02:
        return False, False

    # Resistance is falling (negative slope through peaks)
    pk_slope, _ = _trendline(pk.astype(float), pk_vals)
    if pk_slope >= 0:
        return False, False

    support   = float(tr_vals.mean())
    current   = float(close[-1])
    confirmed = current < support
    forming   = not confirmed and current < support * 1.03
    return forming, confirmed

def _symmetrical_triangle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int = 30
) -> tuple[bool, bool, str]:
    """
    Symmetrical triangle: falling resistance + rising support, converging.
    Returns (forming, confirmed, direction) where direction is 'bullish' or 'bearish'.
    """
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False, "neutral"

    pk_vals = high[-n:][pk]
    tr_vals = low[-n:][tr]

    pk_slope, pk_int = _trendline(pk.astype(float), pk_vals)
    tr_slope, tr_int = _trendline(tr.astype(float), tr_vals)

    # Resistance falling, support rising = symmetrical
    if not (pk_slope < 0 and tr_slope > 0):
        return False, False, "neutral"

    # Projected lines at current bar
    curr_resist = pk_int + pk_slope * n
    curr_supp   = tr_int + tr_slope * n
    current     = float(close[-1])

    if current > curr_resist:
        return False, True, "bullish"
    elif current < curr_supp:
        return False, True, "bearish"
    else:
        return True, False, "neutral"

def _flag(close: np.ndarray, volume: np.ndarray, n_pole: int = 10, n_flag: int = 15
         ) -> tuple[bool, bool, str]:
    """
    Detect bull/bear flag: strong directional move (pole) followed by
    tight consolidation against the move (flag body).
    Returns (forming, confirmed, direction).
    """
    if len(close) < n_pole + n_flag:
        return False, False, "neutral"

    pole    = close[-(n_pole + n_flag):-n_flag]
    flag    = close[-n_flag:]
    pole_ret = (pole[-1] - pole[0]) / pole[0]  # return over the pole

    flag_slope, _ = _trendline(np.arange(n_flag, dtype=float), flag)

    # Bullish flag: pole up, flag drifts slightly down
    if pole_ret > 0.05 and -0.005 >= flag_slope / pole[-1]:
        breakout = float(close[-1]) > float(flag.max())
        return not breakout, breakout, "bullish"

    # Bearish flag: pole down, flag drifts slightly up
    if pole_ret < -0.05 and flag_slope / abs(pole[-1]) > 0:
        breakout = float(close[-1]) < float(flag.min())
        return not breakout, breakout, "bearish"

    return False, False, "neutral"

def _rising_wedge(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 30
) -> tuple[bool, bool]:
    """Rising wedge: both trendlines slope up but converge → bearish."""
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False

    pk_slope, pk_int = _trendline(pk.astype(float), high[-n:][pk])
    tr_slope, tr_int = _trendline(tr.astype(float), low[-n:][tr])

    # Both sloping up, but resistance rising slower than support (converging)
    if not (pk_slope > 0 and tr_slope > 0 and tr_slope > pk_slope):
        return False, False

    support   = tr_int + tr_slope * n
    current   = float(close[-1])
    confirmed = current < support
    forming   = not confirmed
    return forming, confirmed

def _falling_wedge(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 30
) -> tuple[bool, bool]:
    """Falling wedge: both trendlines slope down but converge → bullish."""
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False

    pk_slope, pk_int = _trendline(pk.astype(float), high[-n:][pk])
    tr_slope, tr_int = _trendline(tr.astype(float), low[-n:][tr])

    # Both sloping down, resistance falling faster than support (converging)
    if not (pk_slope < 0 and tr_slope < 0 and pk_slope < tr_slope):
        return False, False

    resistance = pk_int + pk_slope * n
    current    = float(close[-1])
    confirmed  = current > resistance
    forming    = not confirmed
    return forming, confirmed

def _rectangle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int = 30, tol: float = 0.03
) -> tuple[bool, bool, str]:
    """
    Rectangle: price oscillating between horizontal support and resistance.
    Returns (forming, confirmed, direction of breakout).
    """
    pk = _peaks(high[-n:], min_dist=4)
    tr = _troughs(low[-n:], min_dist=4)
    if len(pk) < 2 or len(tr) < 2:
        return False, False, "neutral"

    resistance = float(high[-n:][pk].mean())
    support    = float(low[-n:][tr].mean())
    band       = resistance - support

    if band <= 0:
        return False, False, "neutral"

    # Peaks and troughs should be relatively flat (within tol)
    if (_pct_diff(float(high[-n:][pk].max()), float(high[-n:][pk].min())) > tol or
            _pct_diff(float(low[-n:][tr].max()), float(low[-n:][tr].min())) > tol):
        return False, False, "neutral"

    current = float(close[-1])
    if current > resistance:
        return False, True, "bullish"
    elif current < support:
        return False, True, "bearish"
    else:
        forming = support < current < resistance
        return forming, False, "neutral"


# ── Main detector ──────────────────────────────────────────────────────────────

def detect_chart_patterns(ticker: str, df: pd.DataFrame) -> list[ChartSignal]:
    """
    Run all chart pattern detectors on df.
    df must have lowercase columns: open, high, low, close, volume.
    Returns list of ChartSignal (may be empty).
    """
    if df is None or len(df) < 30:
        return []

    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    signals: list[ChartSignal] = []

    def add(pattern, direction, state, action, reason):
        signals.append(ChartSignal(ticker, pattern, direction, state, action, reason))

    # ── Reversal patterns ──────────────────────────────────────────────────
    forming, confirmed = _double_top(close)
    if confirmed:
        add("Double Top", "bearish", "confirmed", "SELL",
            "Double Top confirmed: price broke below neckline — bearish reversal")
    elif forming:
        add("Double Top", "bearish", "forming", "HOLD",
            "Double Top forming: watch for break below neckline")

    forming, confirmed = _double_bottom(close)
    if confirmed:
        add("Double Bottom", "bullish", "confirmed", "BUY",
            "Double Bottom confirmed: price broke above neckline — bullish reversal")
    elif forming:
        add("Double Bottom", "bullish", "forming", "HOLD",
            "Double Bottom forming: watch for break above neckline")

    forming, confirmed = _head_and_shoulders(close)
    if confirmed:
        add("Head & Shoulders", "bearish", "confirmed", "SELL",
            "H&S confirmed: price broke below neckline — high-reliability bearish reversal")
    elif forming:
        add("Head & Shoulders", "bearish", "forming", "HOLD",
            "H&S forming: right shoulder developing, watch neckline")

    forming, confirmed = _inverse_head_and_shoulders(close)
    if confirmed:
        add("Inverse H&S", "bullish", "confirmed", "BUY",
            "Inverse H&S confirmed: price broke above neckline — high-reliability bullish reversal")
    elif forming:
        add("Inverse H&S", "bullish", "forming", "HOLD",
            "Inverse H&S forming: right shoulder developing, watch neckline")

    forming, confirmed = _rising_wedge(high, low, close)
    if confirmed:
        add("Rising Wedge", "bearish", "confirmed", "SELL",
            "Rising Wedge confirmed: price broke below lower trendline — bearish reversal")
    elif forming:
        add("Rising Wedge", "bearish", "forming", "HOLD",
            "Rising Wedge forming: narrowing upward channel, bearish bias")

    forming, confirmed = _falling_wedge(high, low, close)
    if confirmed:
        add("Falling Wedge", "bullish", "confirmed", "BUY",
            "Falling Wedge confirmed: price broke above upper trendline — bullish reversal")
    elif forming:
        add("Falling Wedge", "bullish", "forming", "HOLD",
            "Falling Wedge forming: narrowing downward channel, bullish bias")

    # ── Continuation patterns ──────────────────────────────────────────────
    forming, confirmed = _ascending_triangle(close, high, low)
    if confirmed:
        add("Ascending Triangle", "bullish", "confirmed", "BUY",
            "Ascending Triangle breakout: price cleared flat resistance — bullish continuation")
    elif forming:
        add("Ascending Triangle", "bullish", "forming", "HOLD",
            "Ascending Triangle forming: higher lows pressing flat resistance")

    forming, confirmed = _descending_triangle(close, high, low)
    if confirmed:
        add("Descending Triangle", "bearish", "confirmed", "SELL",
            "Descending Triangle breakdown: price broke flat support — bearish continuation")
    elif forming:
        add("Descending Triangle", "bearish", "forming", "HOLD",
            "Descending Triangle forming: lower highs pressing flat support")

    forming, confirmed, direction = _symmetrical_triangle(close, high, low)
    if confirmed:
        action = "BUY" if direction == "bullish" else "SELL"
        add("Symmetrical Triangle", direction, "confirmed", action,
            f"Symmetrical Triangle breakout {direction}: price exited the converging channel")
    elif forming:
        add("Symmetrical Triangle", "neutral", "forming", "HOLD",
            "Symmetrical Triangle forming: converging highs and lows, await breakout direction")

    forming, confirmed, direction = _flag(close, volume)
    if confirmed and direction == "bullish":
        add("Bullish Flag", "bullish", "confirmed", "BUY",
            "Bullish Flag breakout: strong pole + tight consolidation + upside breakout")
    elif confirmed and direction == "bearish":
        add("Bearish Flag", "bearish", "confirmed", "SELL",
            "Bearish Flag breakdown: strong downmove + brief consolidation + downside break")
    elif forming and direction == "bullish":
        add("Bullish Flag", "bullish", "forming", "HOLD",
            "Bullish Flag forming: tight consolidation after strong upward move")
    elif forming and direction == "bearish":
        add("Bearish Flag", "bearish", "forming", "HOLD",
            "Bearish Flag forming: tight consolidation after strong downward move")

    forming, confirmed, direction = _rectangle(close, high, low)
    if confirmed:
        action = "BUY" if direction == "bullish" else "SELL"
        add("Rectangle", direction, "confirmed", action,
            f"Rectangle breakout {direction}: price cleared the trading range boundary")
    elif forming:
        add("Rectangle", "neutral", "forming", "HOLD",
            "Rectangle forming: price oscillating between horizontal support and resistance")

    return signals


# ── Screener ───────────────────────────────────────────────────────────────────

def screen_chart_patterns(watchlist: list[str]) -> list[dict]:
    """
    Screen a watchlist for chart patterns.
    Returns confirmed signals first, then forming; sorted by reliability.
    Only returns tickers with at least one detected pattern.
    """
    from data.market_data import get_ohlcv
    from loguru import logger

    results = []
    for ticker in watchlist:
        try:
            df = get_ohlcv(ticker, period="6mo", interval="1d")
            if df is None or df.empty or len(df) < 30:
                continue

            patterns = detect_chart_patterns(ticker, df)
            if not patterns:
                continue

            # Confirmed takes priority over forming; BUY over SELL over HOLD
            action_order   = {"BUY": 0, "SELL": 1, "HOLD": 2}
            state_order    = {"confirmed": 0, "forming": 1}
            patterns.sort(key=lambda p: (state_order[p.state], action_order[p.action]))

            best = patterns[0]
            results.append({
                "ticker":       ticker,
                "pattern":      best.pattern,
                "direction":    best.direction,
                "state":        best.state,
                "action":       best.action,
                "reason":       best.reason,
                "all_patterns": [p.pattern for p in patterns],
            })
            logger.info(
                f"Chart | {ticker:6s} [{best.state:9s}] {best.pattern:25s} → {best.action}"
            )
        except Exception as exc:
            logger.warning(f"Chart pattern scan failed for {ticker}: {exc}")

    # Sort: confirmed first, then by action priority
    results.sort(key=lambda x: (
        0 if x["state"] == "confirmed" else 1,
        {"BUY": 0, "SELL": 1, "HOLD": 2}.get(x["action"], 3)
    ))
    return results


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    print(f"Scanning {len(watchlist)} tickers for chart patterns...\n")
    hits = screen_chart_patterns(watchlist)

    if not hits:
        print("No chart patterns detected.")
    else:
        confirmed = [h for h in hits if h["state"] == "confirmed"]
        forming   = [h for h in hits if h["state"] == "forming"]

        if confirmed:
            print("── CONFIRMED BREAKOUTS (trade now) ──────────────────────")
            for h in confirmed:
                print(f"  {h['ticker']:<8} {h['action']:<5} {h['pattern']:<28} {h['reason'][:55]}")

        if forming:
            print("\n── FORMING PATTERNS (watch list) ────────────────────────")
            for h in forming:
                print(f"  {h['ticker']:<8} {h['action']:<5} {h['pattern']:<28} {h['reason'][:55]}")
