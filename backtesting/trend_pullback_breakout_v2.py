"""
backtesting/trend_pullback_breakout_v2.py — Tunable Trend Pullback Breakout.

Adds two knobs over v1 to fix the "too few signals" problem found in testing:

  PULLBACK_TOL_ATR : how close to EMA20 the pullback must get, measured in ATR.
                     v1 required low <= ema20 exactly (tol=0). Loosening to
                     ~0.5 ATR captures shallow pullbacks that never quite touch.

  BREAKOUT_WINDOW  : how many bars AFTER a qualifying pullback the breakout may
                     occur. v1 demanded pullback + breakout on the SAME bar
                     (window=1). Real setups pull back, THEN break out 1-3 bars
                     later. Widening this is the highest-leverage fix.

Everything else matches v1: EMA20/50 trend filter, volume confirmation,
2R target, stop = safer of pullback-extreme / 1.5 ATR, daily-loss circuit
breaker, MAX_POSITION_SIZE_USD sizing.

Set knobs to v1 values (tol=0.0, window=1) to reproduce the original exactly.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass

EMA_FAST, EMA_SLOW = 20, 50
VOL_AVG_PERIOD, VOL_MULT = 20, 1.5
ATR_PERIOD, ATR_STOP_MULT, RR_TARGET = 14, 1.5, 2.0
POSITION_SIZE_USD = 1000

try:
    import config as _cfg  # noqa
    MAX_POSITION_SIZE_USD = float(getattr(_cfg, "MAX_POSITION_SIZE_USD", 500))
    DAILY_LOSS_LIMIT_USD  = float(getattr(_cfg, "DAILY_LOSS_LIMIT_USD", 100))
except Exception:
    MAX_POSITION_SIZE_USD = float(os.getenv("MAX_POSITION_SIZE_USD", 500))
    DAILY_LOSS_LIMIT_USD  = float(os.getenv("DAILY_LOSS_LIMIT_USD", 100))


def _load_and_compute(symbol: str, period: str = "2y") -> pd.DataFrame:
    df = _yf_dl(symbol, period=period, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()
    if len(df) < EMA_SLOW + 10:
        return pd.DataFrame()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    df["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    df["avgvol"]   = volume.rolling(VOL_AVG_PERIOD).mean()
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()
    return df.dropna()


@dataclass
class Trade:
    symbol: str; direction: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float; stop_price: float
    target_price: float; pnl_pct: float; exit_reason: str


def backtest_symbol(symbol, df=None, *, pullback_tol_atr=0.0, breakout_window=1,
                    allow_shorts=True, period="2y") -> list[Trade]:
    if df is None:
        df = _load_and_compute(symbol, period)
    if df is None or df.empty:
        return []

    trades = []
    in_pos = False; direction = 0
    entry_price = stop_price = target_price = 0.0; entry_date = ""
    cur_day = None; day_pnl_usd = 0.0

    # pending pullback state: bars since a qualifying pullback (per direction)
    long_pb_age = short_pb_age = None   # None = no active pullback

    for i in range(2, len(df)):
        row, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        d = df.index[i].date(); date = str(d)
        if d != cur_day:
            cur_day, day_pnl_usd = d, 0.0

        close = float(row["close"]); emaf = float(row["ema_fast"])
        emas = float(row["ema_slow"]); atr = float(row["atr"])
        vol = float(row["volume"]); avgvol = float(row["avgvol"])
        hi, lo = float(row["high"]), float(row["low"])
        tol = pullback_tol_atr * atr

        if not in_pos:
            up, down = emaf > emas, emaf < emas
            vol_ok = avgvol > 0 and vol > avgvol * VOL_MULT

            # 1) detect/refresh pullback proximity to EMA20
            if up and (lo <= emaf + tol):
                long_pb_age = 0
            elif long_pb_age is not None:
                long_pb_age += 1
            if down and (hi >= emaf - tol):
                short_pb_age = 0
            elif short_pb_age is not None:
                short_pb_age += 1

            # expire stale pullbacks / trend flips
            if not up or (long_pb_age is not None and long_pb_age >= breakout_window):
                long_pb_age = None if not up else long_pb_age
            if not down or (short_pb_age is not None and short_pb_age >= breakout_window):
                short_pb_age = None if not down else short_pb_age

            breakout_up   = close > float(prev["high"])
            breakout_down = close < float(prev["low"])

            long_sig  = up and (long_pb_age is not None) and breakout_up and vol_ok
            short_sig = allow_shorts and down and (short_pb_age is not None) and breakout_down and vol_ok

            if day_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
                long_sig = short_sig = False

            if long_sig:
                in_pos, direction = True, 1
                entry_price, entry_date = close, date
                stop_price = min(min(float(prev["low"]), float(prev2["low"])), close - atr*ATR_STOP_MULT)
                target_price = entry_price + (entry_price - stop_price) * RR_TARGET
                long_pb_age = short_pb_age = None
            elif short_sig:
                in_pos, direction = True, -1
                entry_price, entry_date = close, date
                stop_price = max(max(float(prev["high"]), float(prev2["high"])), close + atr*ATR_STOP_MULT)
                target_price = entry_price - (stop_price - entry_price) * RR_TARGET
                long_pb_age = short_pb_age = None
        else:
            reason = None; exit_px = None
            if direction == 1:
                if lo <= stop_price: exit_px, reason = stop_price, "Stop-loss"
                elif hi >= target_price: exit_px, reason = target_price, f"Take-profit ({RR_TARGET}R)"
            else:
                if hi >= stop_price: exit_px, reason = stop_price, "Stop-loss"
                elif lo <= target_price: exit_px, reason = target_price, f"Take-profit ({RR_TARGET}R)"
            if reason:
                pnl = (exit_px - entry_price) / entry_price * direction
                day_pnl_usd += pnl * POSITION_SIZE_USD
                trades.append(Trade(symbol, "long" if direction==1 else "short",
                    entry_date, date, entry_price, exit_px, stop_price,
                    target_price, pnl, reason))
                in_pos = False
    return trades
