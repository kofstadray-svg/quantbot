"""
utils/indicators.py -- Shared technical indicators.

Single source of truth for indicators used across the screener and exit manager,
so both compute them identically. Currently: Supertrend.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Classic Supertrend(period, multiplier). Price above the line = uptrend;
    a close crossing below flips to downtrend. Clean trend filter + exit signal.

    Returns DataFrame (aligned to close.index) with columns:
      supertrend, direction (+1 up / -1 down), upperband, lowerband.
    """
    h = high.astype(float).reset_index(drop=True)
    l = low.astype(float).reset_index(drop=True)
    c = close.astype(float).reset_index(drop=True)
    n = len(c)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = pd.Series(np.nan, index=range(n))
    final_lower = pd.Series(np.nan, index=range(n))
    st          = pd.Series(np.nan, index=range(n))
    direction   = pd.Series(0, index=range(n), dtype=int)

    # First bar where ATR (and thus the bands) is valid.
    first_valid = int(atr.first_valid_index()) if atr.first_valid_index() is not None else None

    for i in range(n):
        # Warmup: ATR not yet defined -> leave NaN, no usable Supertrend.
        if first_valid is None or i < first_valid or math.isnan(atr.iloc[i]):
            continue

        if i == first_valid:
            # Seed the first valid bar directly from the basic bands.
            final_upper.iloc[i] = upper_basic.iloc[i]
            final_lower.iloc[i] = lower_basic.iloc[i]
            if c.iloc[i] <= upper_basic.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            continue

        if (upper_basic.iloc[i] < final_upper.iloc[i - 1]) or (c.iloc[i - 1] > final_upper.iloc[i - 1]):
            final_upper.iloc[i] = upper_basic.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (lower_basic.iloc[i] > final_lower.iloc[i - 1]) or (c.iloc[i - 1] < final_lower.iloc[i - 1]):
            final_lower.iloc[i] = lower_basic.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        prev_st = st.iloc[i - 1]
        if prev_st == final_upper.iloc[i - 1]:
            if c.iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if c.iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return pd.DataFrame({
        "supertrend": st.values,
        "direction":  direction.values,
        "upperband":  final_upper.values,
        "lowerband":  final_lower.values,
    }, index=close.index)


def supertrend_state(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 10, multiplier: float = 3.0) -> dict:
    """
    Current Supertrend state. Returns dict:
      value, bullish (price>line), direction (+1/-1/0),
      dist_pct (% above/below line), just_flipped (direction changed last bar).
    """
    out = {"value": None, "bullish": None, "direction": 0,
           "dist_pct": None, "just_flipped": False}
    try:
        if close is None or len(close) < period + 2:
            return out
        st = supertrend(high, low, close, period, multiplier)
        last_close = float(close.iloc[-1])
        line = float(st["supertrend"].iloc[-1])
        d    = int(st["direction"].iloc[-1])
        if math.isnan(line):
            return out
        out["value"]     = round(line, 4)
        out["direction"] = d
        out["bullish"]   = last_close > line
        out["dist_pct"]  = round((last_close - line) / line * 100.0, 2) if line else None
        if len(st) >= 2:
            out["just_flipped"] = int(st["direction"].iloc[-1]) != int(st["direction"].iloc[-2])
    except Exception:
        pass
    return out
