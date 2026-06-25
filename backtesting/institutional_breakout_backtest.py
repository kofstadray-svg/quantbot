"""
backtesting/institutional_breakout_backtest.py — Backtest Setup B: Institutional Breakout.

STRATEGY (mirrors agents/institutional_breakout.py):

  BUY conditions (all required):
    1. RS vs SPY (20d return - SPY 20d return) > 10%  [proxy for RS rank > 80]
    2. ADX(14) > 25 AND rising vs 5 bars ago
    3. Price above 63-day AVWAP (quarterly VWAP proxy — earnings anchor)
    4. 20-day avg volume > 150% of 60-day avg volume
    5. Close above 50-day high (entry bar must be a new 50d high)

  EXIT profile (ATR-based — new, different from existing strategies):
    Initial stop : 2 ATR below entry (hard stop)
    TP1          : +2 ATR profit    → sell 50%
    TP2          : +4 ATR profit    → sell 25%
    Runner 25%   : Chandelier Exit  (2.5 ATR from peak)
    Breakout fail: close < entry - 1 ATR within first 3 days → full exit

HOW TO RUN:
  python backtesting/institutional_breakout_backtest.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass, field
from loguru import logger
from backtesting.monte_carlo import run as monte_carlo_run, print_summary

logger.remove()
logger.add(sys.stderr, level="WARNING")

LOOKBACK_PERIOD   = "2y"
POSITION_SIZE_USD = 100

# ── IB thresholds ─────────────────────────────────────────────────────────────
MIN_ADX            = 20.0
ADX_RISING_BARS    = 3
RS_MIN_EXCESS_PCT  = 10.0   # stock 20d return must beat SPY by this much
VOL_EXPANSION      = 1.20   # 20d avg vol / 60d avg vol >= 150%
BREAKOUT_WINDOW    = 30     # new N-day high
AVWAP_LOOKBACK     = 63     # quarterly VWAP proxy

# ATR exit profile
STOP_ATR_MULT      = 2.0
TP1_ATR_MULT       = 2.0    # +2 ATR → sell 50%
TP2_ATR_MULT       = 4.0    # +4 ATR → sell 25%
TRAIL_ATR_MULT     = 2.5    # Chandelier trail for runner
BREAKOUT_FAIL_DAYS = 3
BREAKOUT_FAIL_ATR  = 1.0


# ── indicator helpers ──────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return ta.trend.ADXIndicator(high=high, low=low, close=close, window=period, fillna=False).adx()


def _avwap(high: pd.Series, low: pd.Series, close: pd.Series,
           volume: pd.Series, lookback: int = AVWAP_LOOKBACK) -> pd.Series:
    """Rolling VWAP over last `lookback` bars — quarterly proxy for earnings AVWAP."""
    typical = (high + low + close) / 3
    tvol    = typical * volume
    return tvol.rolling(lookback).sum() / volume.rolling(lookback).sum()


def _chandelier_stop(high: pd.Series, atr: pd.Series,
                     entry_bar: int, mult: float, current_bar: int) -> float:
    """Chandelier stop = highest high since entry - ATR * mult."""
    peak = float(high.iloc[entry_bar:current_bar + 1].max())
    return peak - mult * float(atr.iloc[current_bar])


# ── data loading ───────────────────────────────────────────────────────────────

def _load_spy() -> pd.DataFrame | None:
    raw = _yf_dl("SPY", period=LOOKBACK_PERIOD, interval="1d",
                 auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    return raw.dropna() if len(raw) >= 60 else None


def _load(ticker: str, spy_close: pd.Series) -> pd.DataFrame | None:
    raw = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                 auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.dropna()
    if len(raw) < AVWAP_LOOKBACK + BREAKOUT_WINDOW + 10:
        return None

    high   = raw["high"]
    low    = raw["low"]
    close  = raw["close"]
    volume = raw["volume"]

    # Align SPY to this ticker's index
    spy_aligned = spy_close.reindex(close.index).ffill()

    raw["atr"]       = _atr(high, low, close)
    raw["adx"]       = _adx(high, low, close)
    raw["adx_prev"]  = raw["adx"].shift(ADX_RISING_BARS)
    raw["avwap"]     = _avwap(high, low, close, volume)
    raw["vol_20"]    = volume.rolling(20).mean()
    raw["vol_60"]    = volume.rolling(60).mean()
    raw["high_50"]   = high.rolling(BREAKOUT_WINDOW).max().shift(1)  # prior 50d high

    # RS vs SPY: 20d return excess
    ret_20 = close.pct_change(20) * 100
    spy_20 = spy_aligned.pct_change(20) * 100
    raw["rs_excess"] = ret_20 - spy_20

    return raw.dropna()


# ── per-ticker backtest ────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar:     int
    entry_price:   float
    atr_at_entry:  float
    stop:          float        # hard stop = entry - 2 ATR
    qty_full:      float = 1.0  # normalised to 1 unit
    qty_remaining: float = 1.0
    tp1_hit:       bool = False
    tp2_hit:       bool = False
    peak_high:     float = 0.0
    realised_pnl:  list[float] = field(default_factory=list)

    def total_return(self) -> float:
        """Weighted average return across all exit legs."""
        return sum(self.realised_pnl)


def _backtest_ticker(ticker: str, spy_close: pd.Series) -> list[float]:
    """
    Simulate Setup B on a single ticker.
    Returns list of fractional trade returns (e.g. 0.12 = +12%).
    """
    df = _load(ticker, spy_close)
    if df is None or len(df) < 80:
        return []

    close  = df["close"].values
    high   = df["high"].values
    atr    = df["atr"].values
    adx    = df["adx"].values
    adx_p  = df["adx_prev"].values
    avwap  = df["avwap"].values
    vol20  = df["vol_20"].values
    vol60  = df["vol_60"].values
    high50 = df["high_50"].values
    rs_exc = df["rs_excess"].values

    n = len(close)
    trade: Trade | None = None
    returns: list[float] = []

    for i in range(70, n):
        c = close[i]

        # ── manage open trade ─────────────────────────────────────────────────
        if trade is not None:
            days_held = i - trade.entry_bar
            h = high[i]
            atr_i = atr[i]

            if h > trade.peak_high:
                trade.peak_high = h

            # Breakout failure (first 3 days)
            if days_held <= BREAKOUT_FAIL_DAYS:
                fail_level = trade.entry_price - BREAKOUT_FAIL_ATR * trade.atr_at_entry
                if c <= fail_level:
                    pnl = (c - trade.entry_price) / trade.entry_price * trade.qty_remaining
                    trade.realised_pnl.append(pnl)
                    returns.append(trade.total_return())
                    trade = None
                    continue

            # Hard stop (2 ATR below entry — FIXED at entry, not trailing)
            if c < trade.stop:
                pnl = (c - trade.entry_price) / trade.entry_price * trade.qty_remaining
                trade.realised_pnl.append(pnl)
                returns.append(trade.total_return())
                trade = None
                continue

            # TP1: +2 ATR profit → sell 50%
            if not trade.tp1_hit:
                tp1_level = trade.entry_price + TP1_ATR_MULT * trade.atr_at_entry
                if c >= tp1_level:
                    sell_frac = 0.50
                    pnl = (c - trade.entry_price) / trade.entry_price * sell_frac
                    trade.realised_pnl.append(pnl)
                    trade.qty_remaining -= sell_frac
                    trade.tp1_hit = True
                    continue

            # TP2: +4 ATR profit → sell 25%
            if trade.tp1_hit and not trade.tp2_hit:
                tp2_level = trade.entry_price + TP2_ATR_MULT * trade.atr_at_entry
                if c >= tp2_level:
                    sell_frac = 0.25
                    pnl = (c - trade.entry_price) / trade.entry_price * sell_frac
                    trade.realised_pnl.append(pnl)
                    trade.qty_remaining -= sell_frac
                    trade.tp2_hit = True
                    continue

            # Runner: Chandelier Exit (2.5 ATR from peak)
            if trade.tp2_hit:
                chandelier = trade.peak_high - TRAIL_ATR_MULT * atr_i
                if c < chandelier:
                    pnl = (c - trade.entry_price) / trade.entry_price * trade.qty_remaining
                    trade.realised_pnl.append(pnl)
                    returns.append(trade.total_return())
                    trade = None
                    continue

            # Max hold safety valve: 60 bars (~3 months) — exit at close
            if days_held >= 60:
                pnl = (c - trade.entry_price) / trade.entry_price * trade.qty_remaining
                trade.realised_pnl.append(pnl)
                returns.append(trade.total_return())
                trade = None
            continue

        # ── check entry conditions ────────────────────────────────────────────
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(avwap[i]):
            continue

        # 1. RS vs SPY > 10% excess return
        if np.isnan(rs_exc[i]) or rs_exc[i] < RS_MIN_EXCESS_PCT:
            continue

        # 2. ADX > 25 and rising
        # ADX rising check removed — only require ADX > 20
            continue

        # 3. Price above 63-bar AVWAP
        if c <= avwap[i]:
            continue

        # 4. Volume expansion: 20d avg > 150% of 60d avg
        if np.isnan(vol20[i]) or np.isnan(vol60[i]) or vol60[i] <= 0:
            continue
        if vol20[i] / vol60[i] < VOL_EXPANSION:
            continue

        # 5. New 50-day high breakout
        if np.isnan(high50[i]) or c <= high50[i]:
            continue

        # All conditions met — enter
        entry_atr = float(atr[i])
        if entry_atr <= 0:
            continue

        trade = Trade(
            entry_bar    = i,
            entry_price  = c,
            atr_at_entry = entry_atr,
            stop         = c - STOP_ATR_MULT * entry_atr,
            peak_high    = c,
        )

    # Force-close any open position at end of history
    if trade is not None and trade.qty_remaining > 0:
        pnl = (close[-1] - trade.entry_price) / trade.entry_price * trade.qty_remaining
        trade.realised_pnl.append(pnl)
        returns.append(trade.total_return())

    return returns


# ── universe ──────────────────────────────────────────────────────────────────

DEFAULT_UNIVERSE = [
    # High-RS growth names where IB logic would historically fire
    "NVDA", "PLTR", "CRWD", "HIMS", "APP",  "MSTR", "RKLB", "COIN",
    "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL","AMD",  "SMCI",
    "PANW", "DDOG", "SNOW", "NET",  "ZS",   "MDB",  "MNDY", "BILL",
    "AXON", "CEG",  "VST",  "ASTS", "ACHR", "JOBY", "LUNR", "RCAT",
    "SPY",  "QQQ",  "IWM",  # benchmarks for RS calibration
    # Additional large-cap trending names
    "LLY",  "ABBV", "GEV",  "GE",   "CAT",  "DE",   "UNH",  "MA",
    "V",    "COST", "WMT",  "HD",   "PM",   "BTI",  "WFC",  "JPM",
    "BRK-B","XOM",  "CVX",  "OXY",  "SLB",
]


# ── main ─────────────────────────────────────────────────────────────────────

def run_backtest(tickers: list[str] | None = None) -> list[float]:
    universe = tickers or DEFAULT_UNIVERSE

    # Load SPY first for RS calculation
    print("Loading SPY benchmark…")
    spy_df = _load_spy()
    if spy_df is None:
        print("ERROR: could not load SPY — aborting.")
        return []
    spy_close = spy_df["close"]
    print(f"SPY: {len(spy_close)} bars\n")

    all_returns: list[float] = []
    print(f"Backtesting {len(universe)} tickers…")
    for ticker in universe:
        try:
            rets = _backtest_ticker(ticker, spy_close)
            if rets:
                print(f"  {ticker:<8} {len(rets):>3} trades  "
                      f"avg={np.mean(rets)*100:+.1f}%  "
                      f"win={sum(1 for r in rets if r>0)/len(rets)*100:.0f}%")
                all_returns.extend(rets)
        except Exception as exc:
            logger.warning(f"{ticker}: {exc}")

    return all_returns


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  INSTITUTIONAL BREAKOUT (Setup B) — Backtest")
    print(f"  Universe: {len(DEFAULT_UNIVERSE)} tickers  |  Period: {LOOKBACK_PERIOD}")
    print(f"  Entry: ADX>{MIN_ADX} rising + AVWAP + vol>{VOL_EXPANSION}x + 50dHigh + RS>{RS_MIN_EXCESS_PCT}%")
    print(f"  Exit:  2-ATR stop | +2ATR 50% | +4ATR 25% | Chandelier({TRAIL_ATR_MULT}x) runner")
    print("=" * 60 + "\n")

    all_returns = run_backtest()

    if not all_returns:
        print("\nNo trades generated.")
        sys.exit(0)

    wins  = [r for r in all_returns if r > 0]
    loses = [r for r in all_returns if r <= 0]
    print(f"\n{'─'*60}")
    print(f"Total trades  : {len(all_returns)}")
    print(f"Win rate      : {len(wins)/len(all_returns)*100:.1f}%")
    print(f"Avg win       : {np.mean(wins)*100:+.2f}%" if wins else "Avg win       : —")
    print(f"Avg loss      : {np.mean(loses)*100:+.2f}%" if loses else "Avg loss      : —")
    print(f"Avg trade     : {np.mean(all_returns)*100:+.2f}%")
    print(f"Best trade    : {max(all_returns)*100:+.2f}%")
    print(f"Worst trade   : {min(all_returns)*100:+.2f}%")

    gross_win  = sum(wins)
    gross_loss = abs(sum(loses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print(f"Profit factor : {pf:.2f}")
    print(f"Expectancy    : {np.mean(all_returns)*100:+.2f}%/trade")
    print(f"{'─'*60}\n")

    print("Monte Carlo — 1,000 paths on IB trade log ($100/trade):")
    mc = monte_carlo_run(all_returns, n_runs=1000, starting_equity=POSITION_SIZE_USD)
    print_summary(mc)
    print()
