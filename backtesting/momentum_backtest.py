# -*- coding: utf-8 -*-
"""
backtesting/momentum_backtest.py - Backtest the current momentum screener.

CRITERIA (approximated on daily bars):
  1. Relative Volume  > 5x   (today vs 5-day avg)
  2. Volume           > 2M shares
  3. Gap              > 8%   (open vs prev close)
  4. ATR%             > 5%
  5. EMA 9 > EMA 21
  6. RSI              55–72  (exhaustion hard block above 72)
  7. Near High of Day        (close >= high * 0.98)
  8. Price above VWAP        (approximated: close >= open, bullish day structure)
  -- Float < 30M             (skipped in backtest — static data not historical)
  -- Float Rotation > 50%    (skipped in backtest — static data not historical)

EXIT RULES:
  Stop loss   : -10%
  Take profit : +15%
  RSI exit    : RSI drops below 50 (momentum fading)
  Max hold    : 5 trading days

HOW TO RUN:
  python backtesting/momentum_backtest.py
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
from loguru import logger
from backtesting.monte_carlo import run as monte_carlo_run, print_summary

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_PERIOD   = "2y"
POSITION_SIZE_USD = 100
STOP_LOSS_PCT     = 0.10
TAKE_PROFIT_PCT   = 0.15
MAX_HOLD_DAYS     = 5
MIN_CHECKS        = 6   # out of 8 available (float checks skipped); 6 = realistic threshold

# Universe: small/mid-cap names that actually gap 8%+ and spike volume 5x+
# Large caps (AAPL, MSFT etc.) never trigger these criteria — excluded by design.
UNIVERSE = [
    # Current watchlist — small/mid caps with momentum potential
    "NBIS",  "ALAB",  "TSEM",  "WMG",   "DOCN",
    "AUR",   "GEN",   "SMTC",  "BTSG",  "FIGR",
    "MXL",   "LTH",   "RAL",   "GTX",   "LQDA",
    "EXTR",  "STUB",  "INOD",  "BW",    "PENG",
    "PCT",   "OUST",  "SHLS",  "TBLA",  "SLS",
    "MRAM",  "CADL",  "PIII",  "SLE",   "GBTG",
    # Additional small/mid-cap momentum names
    "TSLA",  "AMD",   "NVDA",  "MRVL",  "CRWD",
    "DDOG",  "PANW",  "PLTR",  "SOFI",  "RIVN",
    "LCID",  "JOBY",  "RKLB",  "ACHR",  "LUNR",
    "IONQ",  "RGTI",  "QUBT",  "SOUN",  "BBAI",
]


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    open_  = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    df = df.copy()
    df["ema9"]     = close.ewm(span=9,  adjust=False).mean()
    df["ema21"]    = close.ewm(span=21, adjust=False).mean()
    df["rsi"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"]  = df["atr"] / close * 100
    df["avg_vol5"] = volume.rolling(5).mean().shift(1)   # yesterday's 5-day avg
    df["rel_vol"]  = volume / df["avg_vol5"]
    df["gap_pct"]  = (open_ - close.shift(1)) / close.shift(1) * 100
    df["near_hod"] = (close >= high * 0.98).astype(int)
    df["above_vwap"] = (close >= open_).astype(int)       # bullish bar ≈ closed above VWAP

    return df.dropna()


def _check_entry(row) -> int:
    """Return number of checks passed (out of 8 available)."""
    def _f(x):
        return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)

    rel_vol  = _f(row["rel_vol"])
    volume   = _f(row["Volume"])
    gap_pct  = _f(row["gap_pct"])
    atr_pct  = _f(row["atr_pct"])
    ema9     = _f(row["ema9"])
    ema21    = _f(row["ema21"])
    rsi      = _f(row["rsi"])
    near_hod = _f(row["near_hod"])
    abv_vwap = _f(row["above_vwap"])

    checks = [
        rel_vol  > 5.0,
        volume   > 2_000_000,
        gap_pct  > 8.0,
        atr_pct  > 5.0,
        ema9     > ema21,
        55 <= rsi <= 72,
        bool(near_hod),
        bool(abv_vwap),
    ]
    return sum(checks)


# ── Per-ticker backtest ───────────────────────────────────────────────────────

def _backtest_ticker(ticker: str) -> list[float]:
    try:
        raw = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                     progress=False, auto_adjust=True, timeout=20)
        if raw is None or len(raw) < 60:
            return []
        df = _build_features(raw)
    except Exception as e:
        logger.warning(f"{ticker}: download error — {e}")
        return []

    close  = df["Close"].squeeze()
    trades: list[float] = []
    i = 0
    idx = df.index.tolist()

    while i < len(idx) - MAX_HOLD_DAYS - 1:
        row = df.loc[idx[i]]

        # ── Entry check ───────────────────────────────────────────────────────
        checks_passed = _check_entry(row)
        if checks_passed < MIN_CHECKS:
            i += 1
            continue

        entry_price = float(close.loc[idx[i + 1]])   # buy next open (approx)
        if entry_price <= 0:
            i += 1
            continue

        stop   = entry_price * (1 - STOP_LOSS_PCT)
        target = entry_price * (1 + TAKE_PROFIT_PCT)
        exit_price = None

        # ── Simulate hold ─────────────────────────────────────────────────────
        for j in range(1, MAX_HOLD_DAYS + 1):
            if i + 1 + j >= len(idx):
                break
            fut = df.loc[idx[i + 1 + j]]
            def _fs(x):
                return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)
            fut_low   = _fs(fut["Low"])
            fut_high  = _fs(fut["High"])
            fut_close = _fs(fut["Close"])
            fut_rsi   = _fs(fut["rsi"])

            if fut_low <= stop:
                exit_price = stop
                break
            if fut_high >= target:
                exit_price = target
                break
            if fut_rsi < 50:       # momentum fading
                exit_price = fut_close
                break
            if j == MAX_HOLD_DAYS:
                exit_price = fut_close

        if exit_price is None:
            i += 1
            continue

        ret = (exit_price - entry_price) / entry_price
        trades.append(ret)
        i += MAX_HOLD_DAYS + 1   # skip past this trade

    return trades


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(universe: list[str] | None = None) -> list[float]:
    tickers = universe or UNIVERSE
    all_trades: list[float] = []

    print(f"\n{'='*60}")
    print(f"  MOMENTUM SCREENER BACKTEST  ({LOOKBACK_PERIOD})")
    print("  Criteria : RelVol>5x  Vol>2M  Gap>8%  ATR>5%  EMA9>21  RSI55-72  NearHOD  VWAP")
    print(f"  Need     : {MIN_CHECKS}/8 checks (float checks excluded from backtest)")
    print(f"  Stop     : -{STOP_LOSS_PCT*100:.0f}%   Target: +{TAKE_PROFIT_PCT*100:.0f}%   Max hold: {MAX_HOLD_DAYS}d")
    print(f"  Universe : {len(tickers)} tickers")
    print(f"{'='*60}\n")

    for ticker in tickers:
        trades = _backtest_ticker(ticker)
        if trades:
            wins   = sum(1 for t in trades if t > 0)
            avg    = np.mean(trades) * 100
            logger.info(
                f"{ticker:<6}  {len(trades):>3} trades  "
                f"win%={wins/len(trades)*100:.0f}%  avg={avg:+.2f}%"
            )
            all_trades.extend(trades)
        else:
            logger.info(f"{ticker:<6}  no signals")

    return all_trades


if __name__ == "__main__":
    trades = run_backtest()

    if not trades:
        print("\n⚠️  No trades generated — criteria may be too strict for historical data.")
        print("   Consider lowering MIN_CHECKS or reducing thresholds.\n")
        sys.exit(0)

    wins      = [t for t in trades if t > 0]
    losses    = [t for t in trades if t <= 0]
    win_rate  = len(wins) / len(trades) * 100
    avg_win   = np.mean(wins)  * 100 if wins   else 0
    avg_loss  = np.mean(losses)* 100 if losses else 0
    avg_trade = np.mean(trades)* 100
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    print(f"\n{'='*60}")
    print("  MOMENTUM BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades  : {len(trades)}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Avg win       : {avg_win:+.2f}%")
    print(f"  Avg loss      : {avg_loss:+.2f}%")
    print(f"  Avg trade     : {avg_trade:+.2f}%")
    print(f"  Expectancy    : {expectancy:+.2f}% per trade")
    print(f"  Best trade    : {max(trades)*100:+.2f}%")
    print(f"  Worst trade   : {min(trades)*100:+.2f}%")
    print(f"{'='*60}\n")

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    print("Running Monte Carlo simulation (1,000 paths)...\n")
    mc = monte_carlo_run(trades, starting_equity=1000.0, n_runs=1000)
    print_summary(mc)
