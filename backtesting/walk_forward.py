"""
backtesting/walk_forward.py -- Walk-forward validation of the stock screener.

Architecture
============
  In-sample  (IS)  : 12 months  -- parameters optimised here
  Out-of-sample (OOS): 3 months -- IS-optimal params applied cold
  Step                : 3 months -- window rolls forward each iteration

  Window 0:  IS [T0       .. T0+12m)   OOS [T0+12m .. T0+15m)
  Window 1:  IS [T0+3m    .. T0+15m)   OOS [T0+15m .. T0+18m)
  Window 2:  IS [T0+6m    .. T0+18m)   OOS [T0+18m .. T0+21m)
  ...continues until data is exhausted.

Parameters swept on IS data
============================
  buy_threshold   : total factor score required to enter (2.5 / 3.0 / 3.5 / 4.0)
  stop_atr_mult   : hard stop = entry - N * ATR(14)      (1.0 / 1.5 / 2.0)
  time_stop_days  : exit if no momentum after N days     (3 / 5 / 7)

  IS objective    : Sharpe ratio of OOS-style trades
  → best param combo applied to the following OOS period

Indicators used  (orthogonal set, matches stock_screener.py v2)
=================================================================
  trend      : golden cross + 50MA slope + ADX(14)
  momentum   : RSI(14)
  volatility : ATR percentile rank (1y rolling)
  liquidity  : relative volume + OBV slope
  rel_str    : stock / SPY 20-day slope

Outputs
========
  • Per-window IS vs OOS metrics table
  • Concatenated OOS equity curve
  • Parameter stability map (how much do best params drift?)
  • IS/OOS degradation ratio (1.0 = no overfitting, <0.5 = severe)

HOW TO RUN:
  python backtesting/walk_forward.py
"""
from __future__ import annotations
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta
from loguru import logger
from typing import Optional

# ---------------------------------------------------------------------------
# Walk-forward config
# ---------------------------------------------------------------------------
# Env-var-overridable for quick validation against limited data:
#   $env:WALK_FORWARD_IS_MONTHS=6 ; python backtesting\walk_forward.py
# With IS=6m + OOS=3m = 9m minimum, two windows fit in ~12 months of yfinance
# data (typical after Yahoo's free-endpoint throttles us). Defaults preserve
# the original 15-month-minimum design (IS=12m + OOS=3m).
TRAIN_MONTHS   = int(os.environ.get("WALK_FORWARD_IS_MONTHS", "12"))
VAL_MONTHS     = int(os.environ.get("WALK_FORWARD_OOS_MONTHS", "3"))
STEP_MONTHS    = int(os.environ.get("WALK_FORWARD_STEP_MONTHS", "3"))
TOTAL_HISTORY  = "4y"          # enough data for several windows
POSITION_USD   = 100
MIN_IS_TRADES  = 10            # skip IS optimisation if too few trades

BUY_THRESHOLDS   = [2.5, 3.0, 3.5, 4.0]
STOP_ATR_MULTS   = [1.0, 1.5, 2.0]
TIME_STOP_DAYS_G = [3, 5, 7]

# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    try:
        return ta.trend.ADXIndicator(high=high, low=low, close=close,
                                      window=period).adx().fillna(20.0)
    except Exception:
        return pd.Series(20.0, index=close.index)


def _obv_slope_series(close: pd.Series, volume: pd.Series,
                      lookback: int = 14) -> pd.Series:
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv[i] = obv[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv[i] = obv[i - 1] - volume.iloc[i]
        else:
            obv[i] = obv[i - 1]
    obv_s  = pd.Series(obv, index=close.index)
    slopes = obv_s.rolling(lookback).apply(
        lambda w: np.polyfit(range(len(w)), w, 1)[0], raw=True
    )
    # Normalise by OBV magnitude to get a direction signal
    return slopes.apply(lambda s: "rising" if s > 0 else ("falling" if s < 0 else "flat"))


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _atr_pct_rank_series(atr: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank of ATR within a 1-year window, 0-100."""
    return atr.rolling(window, min_periods=30).rank(pct=True) * 100


def _factor_score_series(df: pd.DataFrame, spy_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute a continuous factor score matching the SYSTEM_PROMPT rules.
    Returns a float series (roughly -8 to +8).
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Trend ────────────────────────────────────────────────────────────
    ma50   = close.rolling(50).mean()
    ma200  = close.rolling(200).mean()
    golden = (ma50 > ma200).astype(float)
    # 10-bar slope of MA50 as % of price per bar
    ma50_slope = ma50.pct_change(10) * 100 / 10
    adx    = _adx_series(high, low, close)

    trend_score = pd.Series(0.0, index=close.index)
    trend_score += np.where(golden,  0.8, -0.8)
    trend_score += np.where(ma50_slope > 1.0,  0.5,
                   np.where(ma50_slope > 0.0,  0.2,
                                               -0.5))
    trend_score += np.where(adx > 30, 0.5,
                   np.where(adx > 20, 0.2, -0.2))

    # ── Momentum (RSI only) ───────────────────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    mom_score = pd.Series(0.0, index=close.index)
    mom_score += np.where(rsi < 30,  1.5,
                 np.where(rsi < 40,  1.0,
                 np.where(rsi < 50,  0.4,
                 np.where(rsi < 60,  0.0,
                 np.where(rsi < 70, -0.5,
                                    -1.5)))))

    # ── Volatility (ATR percentile) ───────────────────────────────────────
    atr      = _atr_series(high, low, close)
    atr_rank = _atr_pct_rank_series(atr)
    vol_score = pd.Series(0.0, index=close.index)
    vol_score += np.where(atr_rank < 20,  0.8,
                 np.where(atr_rank < 35,  0.5,
                 np.where(atr_rank < 55,  0.0,
                 np.where(atr_rank < 75, -0.4,
                                         -0.8))))

    # ── Liquidity (rel vol + OBV) ─────────────────────────────────────────
    avg_vol  = volume.rolling(20).mean()
    rel_vol  = (volume / avg_vol.replace(0, np.nan)).fillna(1.0)
    obv_slp  = _obv_slope_series(close, volume)

    # RS vs SPY
    ratio    = (close / spy_close.reindex(close.index).ffill()).fillna(method="ffill")
    rs_slope = ratio.rolling(20).apply(
        lambda w: (np.polyfit(range(len(w)), w, 1)[0] / w[0] * 100)
        if len(w) > 1 and w[0] > 0 else 0.0, raw=True
    ).fillna(0.0)

    liq_score = pd.Series(0.0, index=close.index)
    liq_score += np.where(rel_vol > 2.0, 0.8,
                 np.where(rel_vol > 1.5, 0.5,
                 np.where(rel_vol > 1.2, 0.3,
                 np.where(rel_vol < 0.8, -0.5, 0.0))))
    liq_score += np.where(obv_slp == "rising", 0.7,
                 np.where(obv_slp == "falling", -0.7, 0.0))
    liq_score += np.where(rs_slope > 0.5,  0.5,
                 np.where(rs_slope < -0.5, -0.5, 0.0))

    total = (pd.Series(trend_score, index=close.index) +
             mom_score + vol_score + liq_score)
    # Return ATR alongside the score so the caller can attach it to the
    # real (non-copied) DataFrame used in simulation. Previously this assigned
    # df["_atr"] on a throwaway df.copy(), so the column never reached _simulate
    # and stop calculation raised KeyError('_atr').
    return total.fillna(0.0), atr


def _load_ticker(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = _yf_dl(ticker, period=TOTAL_HISTORY, interval="1d",
                    progress=False, auto_adjust=True, timeout=20)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 252 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

@dataclass
class WFTrade:
    ticker:      str
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    pnl_pct:     float
    hold_days:   int
    exit_reason: str
    window_id:   int      = 0
    period:      str      = "IS"   # "IS" or "OOS"


def _simulate(
    ticker: str,
    df: pd.DataFrame,
    score: pd.Series,
    start: date,
    end: date,
    buy_threshold: float,
    stop_atr_mult: float,
    time_stop_days: int,
    window_id: int = 0,
    period: str = "IS",
) -> list[WFTrade]:
    """Simulate trades on df[start:end] using given parameters."""
    mask  = (df.index.date >= start) & (df.index.date < end)
    sub   = df.loc[mask].copy()
    sub_s = score.loc[mask]

    if len(sub) < 5:
        return []

    trades = []
    in_pos = False
    entry_price = entry_atr = 0.0
    entry_idx = 0
    entry_date = ""

    for i in range(1, len(sub)):
        idx   = sub.index[i]
        s     = float(sub_s.iloc[i])
        price = float(sub["close"].iloc[i])
        if "_atr" in sub.columns:
            _a = sub["_atr"].iloc[i]
            atr = float(_a) if not np.isnan(_a) else 0.0
        else:
            atr = 0.0
        dt    = str(idx.date())

        if not in_pos:
            if s >= buy_threshold:
                in_pos      = True
                entry_price = price
                entry_atr   = atr if atr > 0 else price * 0.02
                entry_idx   = i
                entry_date  = dt
        else:
            hold = i - entry_idx
            dd   = (price - entry_price) / entry_price
            reason = None

            # Hard stop: 1 ATR (or multiplier)
            stop_level = entry_price - stop_atr_mult * entry_atr
            if price < stop_level:
                reason = f"hard_stop({stop_atr_mult}ATR)"

            # Time stop
            elif hold >= time_stop_days:
                rsi_val = float(
                    ta.momentum.RSIIndicator(sub["close"].iloc[:i+1], window=14)
                    .rsi().iloc[-1]
                )
                ema5 = float(sub["close"].iloc[max(0, i-4):i+1].mean())
                if rsi_val < 50 and price < ema5:
                    reason = f"time_stop({time_stop_days}d)"

            # Score reversal exit
            if reason is None and s < -1.0:
                reason = "score_exit"

            if reason:
                trades.append(WFTrade(
                    ticker=ticker, entry_date=entry_date, exit_date=dt,
                    entry_price=entry_price, exit_price=price,
                    pnl_pct=(price - entry_price) / entry_price,
                    hold_days=hold, exit_reason=reason,
                    window_id=window_id, period=period,
                ))
                in_pos = False
    return trades


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

@dataclass
class WFWindow:
    id:          int
    is_start:    date
    is_end:      date
    oos_start:   date
    oos_end:     date
    best_params: dict      = field(default_factory=dict)
    is_sharpe:   float     = 0.0
    oos_sharpe:  float     = 0.0
    is_trades:   int       = 0
    oos_trades:  int       = 0
    is_mean:     float     = 0.0
    oos_mean:    float     = 0.0


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def _build_windows(first_date: date, last_date: date) -> list[WFWindow]:
    windows = []
    wid = 0
    cursor = first_date
    while True:
        is_start  = cursor
        is_end    = cursor + relativedelta(months=TRAIN_MONTHS)
        oos_start = is_end
        oos_end   = oos_start + relativedelta(months=VAL_MONTHS)
        if oos_end > last_date:
            break
        windows.append(WFWindow(wid, is_start, is_end, oos_start, oos_end))
        cursor += relativedelta(months=STEP_MONTHS)
        wid += 1
    return windows


def run_walk_forward(tickers: list[str], verbose: bool = True) -> dict:
    """
    Run walk-forward validation across all tickers.
    Returns dict with windows, all OOS trades, equity curve, summary stats.
    """
    print("\nLoading SPY (benchmark) + regime calendar...")
    spy_raw = yf.download("SPY", period=TOTAL_HISTORY, interval="1d",
                          progress=False, auto_adjust=True, timeout=20)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_close = spy_raw["close"].squeeze()

    # Build regime calendar for OOS regime segmentation
    regime_calendar: dict = {}
    try:
        from backtesting.engine import build_regime_calendar
        regime_calendar = build_regime_calendar()
        print(f"  Regime calendar: {len(regime_calendar)} dates labeled")
    except Exception as _re:
        print(f"  Regime calendar unavailable: {_re}")

    # Load and score all tickers upfront
    print(f"Loading {len(tickers)} tickers...")
    ticker_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for ticker in tickers:
        df = _load_ticker(ticker)
        if df is None:
            continue
        try:
            score, atr = _factor_score_series(df, spy_close)
            df = df.copy()
            df["_atr"] = atr.reindex(df.index)
            ticker_data[ticker] = (df, score)
        except Exception as exc:
            logger.warning(f"  {ticker} indicator error: {exc}")
    print(f"  {len(ticker_data)} tickers usable")

    # Determine date range from data
    all_dates = [df.index.date.min() for df, _ in ticker_data.values()]
    all_ends  = [df.index.date.max() for df, _ in ticker_data.values()]
    first_date = max(all_dates)   # conservative: all tickers must have data
    last_date  = min(all_ends)
    windows    = _build_windows(first_date, last_date)

    if not windows:
        print("ERROR: Not enough history for even one walk-forward window.")
        print(f"  Available: {first_date} to {last_date}")
        print(f"  Need at least {TRAIN_MONTHS + VAL_MONTHS} months")
        return {}

    print(f"\nDate range: {first_date} to {last_date}")
    print(f"Walk-forward windows: {len(windows)}")
    print(f"  IS={TRAIN_MONTHS}m  OOS={VAL_MONTHS}m  step={STEP_MONTHS}m")
    print(f"  Buy thresholds: {BUY_THRESHOLDS}")
    print(f"  Stop ATR mults: {STOP_ATR_MULTS}")
    print(f"  Time stop days: {TIME_STOP_DAYS_G}\n")

    param_grid = list(itertools.product(
        BUY_THRESHOLDS, STOP_ATR_MULTS, TIME_STOP_DAYS_G
    ))

    all_oos_trades: list[WFTrade] = []

    for w in windows:
        print(f"Window {w.id}  IS:[{w.is_start}..{w.is_end})  "
              f"OOS:[{w.oos_start}..{w.oos_end})", end="  ", flush=True)

        # ── IS optimisation ───────────────────────────────────────────────
        best_sharpe = -999.0
        best_params = {"buy_threshold": 3.0, "stop_atr_mult": 1.5,
                       "time_stop_days": 5}
        best_is_trades: list[WFTrade] = []

        for (bt, sm, td) in param_grid:
            is_trades: list[WFTrade] = []
            for ticker, (df, score) in ticker_data.items():
                is_trades.extend(_simulate(
                    ticker, df, score,
                    w.is_start, w.is_end,
                    buy_threshold=bt, stop_atr_mult=sm, time_stop_days=td,
                    window_id=w.id, period="IS",
                ))
            if len(is_trades) < MIN_IS_TRADES:
                continue
            sh = _sharpe([t.pnl_pct for t in is_trades])
            if sh > best_sharpe:
                best_sharpe   = sh
                best_params   = {"buy_threshold": bt, "stop_atr_mult": sm,
                                 "time_stop_days": td}
                best_is_trades = is_trades

        w.best_params = best_params
        w.is_sharpe   = best_sharpe
        w.is_trades   = len(best_is_trades)
        w.is_mean     = float(np.mean([t.pnl_pct for t in best_is_trades])) \
                        if best_is_trades else 0.0

        # ── OOS application ───────────────────────────────────────────────
        oos_trades: list[WFTrade] = []
        for ticker, (df, score) in ticker_data.items():
            oos_trades.extend(_simulate(
                ticker, df, score,
                w.oos_start, w.oos_end,
                window_id=w.id, period="OOS", **best_params,
            ))

        w.oos_sharpe = _sharpe([t.pnl_pct for t in oos_trades])
        w.oos_trades = len(oos_trades)
        w.oos_mean   = float(np.mean([t.pnl_pct for t in oos_trades])) \
                       if oos_trades else 0.0

        all_oos_trades.extend(oos_trades)

        print(f"IS={w.is_trades}tr Sh={w.is_sharpe:.2f}  "
              f"OOS={w.oos_trades}tr Sh={w.oos_sharpe:.2f}  "
              f"params={best_params}")

    return {
        "windows":          windows,
        "oos_trades":       all_oos_trades,
        "tickers":          list(ticker_data.keys()),
        "regime_calendar":  regime_calendar,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _equity_curve(trades: list[WFTrade], start_equity: float = 1000.0) -> list[float]:
    equity = [start_equity]
    for t in sorted(trades, key=lambda x: x.exit_date):
        equity.append(equity[-1] * (1 + t.pnl_pct * POSITION_USD / equity[-1]))
    return equity


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0]; mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


def print_report(result: dict, show_regime: bool = True, show_mc: bool = True) -> None:
    if not result:
        return
    windows    = result["windows"]
    oos_trades = result["oos_trades"]

    print(f"""
{'='*70}
  WALK-FORWARD VALIDATION REPORT
  IS={TRAIN_MONTHS}m  OOS={VAL_MONTHS}m  step={STEP_MONTHS}m
  Orthogonal factors: trend · momentum · volatility · liquidity · rel_strength
{'='*70}

  PER-WINDOW SUMMARY
  {'Win':>3}  {'IS start':>10}  {'OOS start':>10}  {'IS Sh':>6}  {'OOS Sh':>6}  {'IS tr':>5}  {'OOS tr':>5}  {'IS mean':>8}  {'OOS mean':>9}  Best params
  {'-'*110}""")

    for w in windows:
        bp = w.best_params
        print(
            f"  {w.id:>3}  {str(w.is_start):>10}  {str(w.oos_start):>10}  "
            f"{w.is_sharpe:>6.2f}  {w.oos_sharpe:>6.2f}  "
            f"{w.is_trades:>5}  {w.oos_trades:>5}  "
            f"{w.is_mean:>+8.2%}  {w.oos_mean:>+9.2%}  "
            f"buy>={bp.get('buy_threshold','-')}  "
            f"stop={bp.get('stop_atr_mult','-')}ATR  "
            f"time={bp.get('time_stop_days','-')}d"
        )

    if not oos_trades:
        print("\n  No OOS trades generated.")
        return

    oos_returns = [t.pnl_pct for t in oos_trades]
    wins   = [r for r in oos_returns if r > 0]
    losses = [r for r in oos_returns if r <= 0]
    equity = _equity_curve(oos_trades)
    mdd    = _max_drawdown(equity)

    avg_is_sh  = float(np.mean([w.is_sharpe  for w in windows if w.is_trades  > 0]))
    avg_oos_sh = float(np.mean([w.oos_sharpe for w in windows if w.oos_trades > 0]))
    degradation = avg_oos_sh / avg_is_sh if avg_is_sh > 0 else 0.0

    # Parameter stability
    bt_vals = [w.best_params.get("buy_threshold", 0) for w in windows]
    sm_vals = [w.best_params.get("stop_atr_mult", 0) for w in windows]
    td_vals = [w.best_params.get("time_stop_days", 0) for w in windows]

    win_rate = len(wins) / len(oos_returns) if oos_returns else 0
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss else float("inf")
    exp      = win_rate * avg_win + (1 - win_rate) * avg_loss

    exit_counts: dict[str, int] = {}
    for t in oos_trades:
        k = t.exit_reason.split("(")[0]
        exit_counts[k] = exit_counts.get(k, 0) + 1

    print(f"""
  CONSOLIDATED OOS METRICS  ({len(windows)} windows, {len(result['tickers'])} tickers)
  {'─'*60}
  Total OOS trades  : {len(oos_trades)}
  Win rate          : {win_rate:.1%}
  Avg win           : {avg_win:+.2%}
  Avg loss          : {avg_loss:+.2%}
  Risk/Reward       : {rr:.2f}:1
  Expectancy/trade  : {exp:+.3%}
  OOS Sharpe (avg)  : {avg_oos_sh:.2f}
  Max drawdown      : {mdd:.1%}
  Final equity      : ${equity[-1]:.0f}  (started $1000)

  IS vs OOS DEGRADATION
  {'─'*60}
  Avg IS  Sharpe    : {avg_is_sh:.2f}
  Avg OOS Sharpe    : {avg_oos_sh:.2f}
  Degradation ratio : {degradation:.2f}
    1.00 = IS performance fully replicates in OOS (no overfitting)
    0.50 = OOS is half as good as IS (moderate overfitting)
    0.00 = OOS performance collapses (severe overfitting)

  PARAMETER STABILITY  (instability = overfitting signal)
  {'─'*60}
  buy_threshold  : values={sorted(set(bt_vals))}  std={np.std(bt_vals):.2f}
  stop_atr_mult  : values={sorted(set(sm_vals))}  std={np.std(sm_vals):.2f}
  time_stop_days : values={sorted(set(td_vals))}  std={np.std(td_vals):.2f}

  Exit breakdown  : {', '.join(f"{k}={v}" for k, v in exit_counts.items())}
  OOS equity curve: {' '.join(f"{e:.0f}" for e in equity[::max(1,len(equity)//15)])}
{'='*70}
""")

    # Actionable conclusion
    if degradation >= 0.7:
        verdict = "ROBUST -- OOS performance tracks IS well. Parameters are stable."
    elif degradation >= 0.4:
        verdict = "MODERATE -- Some IS->OOS degradation. Consider wider param ranges."
    else:
        verdict = "OVERFIT -- OOS collapses vs IS. Simplify the model or reduce params."
    print(f"  VERDICT: {verdict}\n")

    # ── Regime segmentation across all OOS trades ──────────────────────────────
    if show_regime and result.get("regime_calendar"):
        from backtesting.engine import print_regime_breakdown, SimTrade, label_regime
        # Convert WFTrade → SimTrade (minimal mapping for regime reporting)
        from datetime import date as _date
        cal = result["regime_calendar"]
        sim_trades = []
        for t in oos_trades:
            entry_d = _date.fromisoformat(t.entry_date)
            regime  = label_regime(entry_d, cal)
            sim_trades.append(SimTrade(
                ticker=t.ticker, entry_date=t.entry_date, exit_date=t.exit_date,
                entry_price=t.entry_price, exit_price=t.exit_price,
                actual_entry=t.entry_price, actual_exit=t.exit_price,
                raw_pnl_pct=t.pnl_pct, net_pnl_pct=t.pnl_pct,
                fill_fraction=1.0, latency_bars=0, latency_cost=0.0,
                spread_cost=0.0, hold_days=t.hold_days, exit_reason=t.exit_reason,
                regime=regime, window_id=t.window_id, period=t.period,
            ))
        print_regime_breakdown(sim_trades)
        print()

    # ── Block-bootstrap Monte Carlo on OOS returns ─────────────────────────────
    if show_mc and oos_returns:
        from backtesting.monte_carlo import run_block, print_summary as _mc_print
        print("\n  BLOCK BOOTSTRAP MONTE CARLO (1,000 paths on OOS returns)\n")
        mc = run_block(oos_returns, starting_equity=1000, n_runs=1000, block_size=8)
        _mc_print(mc, label="Walk-Forward OOS")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    tickers = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=20) + get_dow_watchlist()
    ))

    print("\nWalk-Forward Validation")
    print(f"Tickers: {len(tickers)}  |  IS:{TRAIN_MONTHS}m  OOS:{VAL_MONTHS}m  step:{STEP_MONTHS}m")

    result = run_walk_forward(tickers, verbose=True)
    print_report(result)
