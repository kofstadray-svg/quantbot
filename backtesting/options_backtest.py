# -*- coding: utf-8 -*-
"""
backtesting/options_backtest.py — Backtest the options layer on top of the
momentum strategy.

WHAT IT MODELS:
  CALLS  — Fired on every momentum BUY signal (proxy for STRONG BUY).
           Buy a call: 35 DTE, strike +2% OTM, sized at $50 notional.
           Exit when underlying hits +15% target, -10% stop, or 5-day max hold.

  PUTS   — Fired on BREAKDOWN signals (RSI<35, gap down>3%, EMA9<21, RelVol>2x).
           Buy a put: 35 DTE, strike -2% OTM, sized at $50 notional.
           Exit when underlying falls -10%, rises +5% (stop), or 5-day max hold.

OPTION PRICING:
  Uses Black-Scholes with IV estimated from 20-day realized volatility × 1.2
  (IV typically runs ~20% above realized vol for momentum / small-cap names).
  Risk-free rate: 5%.

HOW TO RUN:
  python backtesting/options_backtest.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from loguru import logger
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_PERIOD    = "2y"
STOCK_NOTIONAL     = 100      # $ per stock trade
OPTIONS_NOTIONAL   = 50       # $ per option trade
CALL_STRIKE_MULT   = 1.02     # 2% OTM call
PUT_STRIKE_MULT    = 0.98     # 2% OTM put
DTE                = 35       # days to expiration at entry
RISK_FREE_RATE     = 0.05     # 5% annualised
IV_MULTIPLIER      = 1.2      # IV = realized vol × 1.2
STOP_LOSS_PCT      = 0.10     # -10% stop on underlying
CALL_TARGET_PCT    = 0.15     # +15% take-profit on underlying
PUT_TARGET_PCT     = 0.10     # -10% take-profit on underlying (price falls)
PUT_STOP_PCT       = 0.05     # +5% stop on underlying for puts
MAX_HOLD_DAYS      = 5
MIN_CHECKS         = 6        # same as momentum backtest
MIN_OPTION_PRICE   = 0.10     # skip penny options — B-S precision breaks down below this
MAX_QTY            = 10       # cap contracts so one trade can't blow up the portfolio

UNIVERSE = [
    "NBIS",  "ALAB",  "TSEM",  "WMG",   "DOCN",
    "AUR",   "GEN",   "SMTC",  "BTSG",  "FIGR",
    "MXL",   "LTH",   "RAL",   "GTX",   "LQDA",
    "EXTR",  "STUB",  "INOD",  "BW",    "PENG",
    "PCT",   "OUST",  "SHLS",  "TBLA",  "SLS",
    "MRAM",  "CADL",  "PIII",  "SLE",   "GBTG",
    "TSLA",  "AMD",   "NVDA",  "MRVL",  "CRWD",
    "DDOG",  "PANW",  "PLTR",  "SOFI",  "RIVN",
    "LCID",  "JOBY",  "RKLB",  "ACHR",  "LUNR",
    "IONQ",  "RGTI",  "QUBT",  "SOUN",  "BBAI",
]


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Black-Scholes option price. T in years."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _estimate_iv(close: pd.Series) -> float:
    """Annualised realized vol from last 20 days × IV_MULTIPLIER."""
    if len(close) < 21:
        return 0.40
    log_ret = np.log(close.iloc[-20:] / close.iloc[-21:-1].values)
    realized = float(log_ret.std() * math.sqrt(252))
    return max(0.15, min(realized * IV_MULTIPLIER, 1.50))   # clamp 15–150%


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    open_  = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    df = df.copy()
    df["ema9"]      = close.ewm(span=9,  adjust=False).mean()
    df["ema21"]     = close.ewm(span=21, adjust=False).mean()
    df["rsi"]       = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"]       = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"]   = df["atr"] / close * 100
    df["avg_vol5"]  = volume.rolling(5).mean().shift(1)
    df["rel_vol"]   = volume / df["avg_vol5"]
    df["gap_pct"]   = (open_ - close.shift(1)) / close.shift(1) * 100
    df["near_hod"]  = (close >= high * 0.98).astype(int)
    df["abv_vwap"]  = (close >= open_).astype(int)
    return df.dropna()


def _fs(x):
    return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)


def _call_checks(row) -> int:
    return sum([
        _fs(row["rel_vol"])  > 5.0,
        _fs(row["Volume"])   > 2_000_000,
        _fs(row["gap_pct"])  > 8.0,
        _fs(row["atr_pct"])  > 5.0,
        _fs(row["ema9"])     > _fs(row["ema21"]),
        55 <= _fs(row["rsi"]) <= 72,
        bool(_fs(row["near_hod"])),
        bool(_fs(row["abv_vwap"])),
    ])


def _breakdown_check(row) -> bool:
    return (
        _fs(row["rsi"])     < 35   and
        _fs(row["ema9"])    < _fs(row["ema21"]) and
        _fs(row["gap_pct"]) < -3.0 and
        _fs(row["rel_vol"]) > 2.0  and
        _fs(row["atr_pct"]) > 3.0
    )


# ── Per-ticker backtest ───────────────────────────────────────────────────────

def _backtest_ticker(ticker: str) -> tuple[list[float], list[float], list[float]]:
    """
    Returns (stock_returns, call_returns, put_returns).
    All returns are fractional (0.15 = +15%).
    """
    try:
        raw = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                     progress=False, auto_adjust=True, timeout=20)
        if raw is None or len(raw) < 60:
            return [], [], []
        df = _build_features(raw)
    except Exception as e:
        logger.warning(f"{ticker}: download error — {e}")
        return [], [], []

    close = df["Close"].squeeze()
    idx   = df.index.tolist()

    stock_trades: list[float] = []
    call_trades:  list[float] = []
    put_trades:   list[float] = []

    i = 0
    while i < len(idx) - MAX_HOLD_DAYS - 1:
        row = df.loc[idx[i]]

        # ── CALL: fires on momentum BUY signal ────────────────────────────────
        if _call_checks(row) >= MIN_CHECKS:
            entry_price = _fs(close.loc[idx[i + 1]])
            if entry_price > 0:
                iv        = _estimate_iv(close.iloc[:i + 1])
                K_call    = entry_price * CALL_STRIKE_MULT
                T_entry   = DTE / 365
                opt_entry = _bs_price(entry_price, K_call, T_entry, RISK_FREE_RATE, iv, "call")

                stock_stop   = entry_price * (1 - STOP_LOSS_PCT)
                stock_target = entry_price * (1 + CALL_TARGET_PCT)

                exit_price   = None
                days_held    = 0

                for j in range(1, MAX_HOLD_DAYS + 1):
                    if i + 1 + j >= len(idx):
                        break
                    fut       = df.loc[idx[i + 1 + j]]
                    fut_low   = _fs(fut["Low"])
                    fut_high  = _fs(fut["High"])
                    fut_close = _fs(fut["Close"])
                    fut_rsi   = _fs(fut["rsi"])

                    if fut_low <= stock_stop:
                        exit_price = stock_stop; days_held = j; break
                    if fut_high >= stock_target:
                        exit_price = stock_target; days_held = j; break
                    if fut_rsi < 50:
                        exit_price = fut_close; days_held = j; break
                    if j == MAX_HOLD_DAYS:
                        exit_price = fut_close; days_held = j

                if exit_price and opt_entry >= MIN_OPTION_PRICE:
                    T_exit   = max(0.001, (DTE - days_held) / 365)
                    opt_exit = max(0.0, _bs_price(exit_price, K_call, T_exit, RISK_FREE_RATE, iv, "call"))
                    qty      = min(MAX_QTY, max(1, math.floor(OPTIONS_NOTIONAL / (opt_entry * 100))))
                    call_pl  = (opt_exit - opt_entry) * qty * 100
                    call_ret = max(-1.0, call_pl / OPTIONS_NOTIONAL)
                    call_trades.append(call_ret)

                    stock_ret = (exit_price - entry_price) / entry_price
                    stock_trades.append(stock_ret)

            i += MAX_HOLD_DAYS + 1
            continue

        # ── PUT: fires on breakdown signal ────────────────────────────────────
        if _breakdown_check(row):
            entry_price = _fs(close.loc[idx[i + 1]])
            if entry_price > 0:
                iv        = _estimate_iv(close.iloc[:i + 1])
                K_put     = entry_price * PUT_STRIKE_MULT
                T_entry   = DTE / 365
                opt_entry = _bs_price(entry_price, K_put, T_entry, RISK_FREE_RATE, iv, "put")

                put_target = entry_price * (1 - PUT_TARGET_PCT)   # price falls here → profit
                put_stop   = entry_price * (1 + PUT_STOP_PCT)      # price rises → cut loss

                exit_price = None
                days_held  = 0

                for j in range(1, MAX_HOLD_DAYS + 1):
                    if i + 1 + j >= len(idx):
                        break
                    fut       = df.loc[idx[i + 1 + j]]
                    fut_low   = _fs(fut["Low"])
                    fut_high  = _fs(fut["High"])
                    fut_close = _fs(fut["Close"])

                    if fut_low <= put_target:
                        exit_price = put_target; days_held = j; break
                    if fut_high >= put_stop:
                        exit_price = put_stop; days_held = j; break
                    if j == MAX_HOLD_DAYS:
                        exit_price = fut_close; days_held = j

                if exit_price and opt_entry >= MIN_OPTION_PRICE:
                    T_exit   = max(0.001, (DTE - days_held) / 365)
                    opt_exit = max(0.0, _bs_price(exit_price, K_put, T_exit, RISK_FREE_RATE, iv, "put"))
                    qty      = min(MAX_QTY, max(1, math.floor(OPTIONS_NOTIONAL / (opt_entry * 100))))
                    put_pl   = (opt_exit - opt_entry) * qty * 100
                    put_ret  = max(-1.0, put_pl / OPTIONS_NOTIONAL)
                    put_trades.append(put_ret)

        i += 1

    return stock_trades, call_trades, put_trades


# ── Summary helpers ───────────────────────────────────────────────────────────

def _stats(trades: list[float], label: str, notional: float = OPTIONS_NOTIONAL) -> None:
    if not trades:
        print(f"  {label}: no trades")
        return
    wins     = [t for t in trades if t > 0]
    losses   = [t for t in trades if t <= 0]
    wr       = len(wins) / len(trades) * 100
    avg_w    = np.mean(wins)   * 100 if wins   else 0
    avg_l    = np.mean(losses) * 100 if losses else 0
    exp_pct  = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)
    exp_usd  = exp_pct / 100 * notional
    avg_usd  = np.mean(trades) * notional
    print(f"  {label}")
    print(f"    Trades      : {len(trades)}")
    print(f"    Win rate    : {wr:.1f}%")
    print(f"    Avg win     : {avg_w:+.2f}%")
    print(f"    Avg loss    : {avg_l:+.2f}%")
    print(f"    Expectancy  : {exp_pct:+.2f}% per trade  (${exp_usd:+.2f} on ${notional:.0f} notional)")
    print(f"    Avg P&L/trade: ${avg_usd:+.2f}")
    print(f"    Best        : {max(trades)*100:+.2f}%   Worst: {min(trades)*100:+.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*62}")
    print(f"  OPTIONS LAYER BACKTEST  ({LOOKBACK_PERIOD})")
    print("  Calls : BUY signal → +2% OTM call, 35 DTE, $50 notional")
    print("  Puts  : BREAKDOWN  → -2% OTM put,  35 DTE, $50 notional")
    print(f"  IV    : 20-day realized vol × {IV_MULTIPLIER}  |  r={RISK_FREE_RATE*100:.0f}%")
    print(f"  Universe : {len(UNIVERSE)} tickers")
    print(f"{'='*62}\n")

    all_stock:  list[float] = []
    all_calls:  list[float] = []
    all_puts:   list[float] = []

    for ticker in UNIVERSE:
        s, c, p = _backtest_ticker(ticker)
        if c or p:
            logger.info(
                f"{ticker:<6}  stock={len(s)}  calls={len(c)}  puts={len(p)}"
            )
        all_stock.extend(s)
        all_calls.extend(c)
        all_puts.extend(p)

    print(f"\n{'='*62}")
    print("  RESULTS")
    print(f"{'='*62}")
    _stats(all_stock, "Stock trades (baseline)",          notional=STOCK_NOTIONAL)
    print()
    _stats(all_calls, "Call options (on BUY signals)",    notional=OPTIONS_NOTIONAL)
    print()
    _stats(all_puts,  "Put options  (on BREAKDOWN signals)", notional=OPTIONS_NOTIONAL)

    # ── Combined: stock + call on every BUY signal ────────────────────────────
    if all_stock and all_calls and len(all_stock) == len(all_calls):
        combined = []
        for s, c in zip(all_stock, all_calls):
            # $100 stock + $50 call = $150 total deployed
            combined_ret = (s * STOCK_NOTIONAL + c * OPTIONS_NOTIONAL) / (STOCK_NOTIONAL + OPTIONS_NOTIONAL)
            combined.append(combined_ret)
        print()
        _stats(combined, "Combined  (stock + call, $150 total)", notional=STOCK_NOTIONAL + OPTIONS_NOTIONAL)

    # ── Fixed-dollar simulation ───────────────────────────────────────────────
    # Options are traded with a fixed notional ($50/trade), NOT compounded.
    # The standard % Monte Carlo is wrong here: it compounds returns as if
    # you reinvest all winnings, producing ruin even with positive expectancy.
    # This simulation boots 1,000 paths of N trades each drawn (with replacement)
    # from the observed dollar P&L pool and tracks cumulative $ gain.
    rng = np.random.default_rng(42)

    def _dollar_sim(trades_pct: list[float], notional: float,
                    n_runs: int = 1_000, label: str = "") -> None:
        if not trades_pct:
            return
        dollar_pl = np.array(trades_pct) * notional   # convert to $ P&L per trade
        n_trades  = len(dollar_pl)
        paths     = rng.choice(dollar_pl, size=(n_runs, n_trades), replace=True)
        cumulative = paths.cumsum(axis=1)              # $ profit after each trade
        final      = cumulative[:, -1]
        pct_profitable = (final > 0).mean() * 100
        median_gain    = np.median(final)
        p5             = np.percentile(final, 5)
        p95            = np.percentile(final, 95)
        max_drawdown   = (cumulative - np.maximum.accumulate(cumulative, axis=1)).min(axis=1)
        avg_dd         = np.mean(max_drawdown)

        print(f"\n{'='*62}")
        print(f"  FIXED-$ SIMULATION — {label}")
        print(f"  {n_trades} trades resampled × {n_runs} paths  |  ${notional:.0f}/trade")
        print(f"{'='*62}")
        print(f"  % paths profitable : {pct_profitable:.1f}%")
        print(f"  Median total gain  : ${median_gain:+.2f}")
        print(f"  5th / 95th pct     : ${p5:+.2f} / ${p95:+.2f}")
        print(f"  Avg max drawdown   : ${avg_dd:+.2f}")
        print(f"  Expected total gain: ${float(np.mean(final)):+.2f}  "
              f"(${float(np.mean(dollar_pl)):+.2f}/trade × {n_trades} trades)")

    _dollar_sim(all_calls, OPTIONS_NOTIONAL, label="Calls only")
    _dollar_sim(all_puts,  OPTIONS_NOTIONAL, label="Puts only")

    if all_stock and all_calls and len(all_stock) == len(all_calls):
        combined_dollar = [
            s * STOCK_NOTIONAL + c * OPTIONS_NOTIONAL
            for s, c in zip(all_stock, all_calls)
        ]
        rng2 = np.random.default_rng(42)
        arr  = np.array(combined_dollar)
        n    = len(arr)
        paths = rng2.choice(arr, size=(1_000, n), replace=True).cumsum(axis=1)
        final = paths[:, -1]
        print(f"\n{'='*62}")
        print("  FIXED-$ SIMULATION — Combined (stock $100 + call $50)")
        print(f"{'='*62}")
        print(f"  % paths profitable : {(final > 0).mean()*100:.1f}%")
        print(f"  Median total gain  : ${np.median(final):+.2f}")
        print(f"  5th / 95th pct     : ${np.percentile(final,5):+.2f} / ${np.percentile(final,95):+.2f}")
        print(f"  Expected total gain: ${float(np.mean(final)):+.2f}  "
              f"(${float(np.mean(arr)):+.2f}/trade × {n} trades)")
