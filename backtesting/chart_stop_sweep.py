"""
backtesting/chart_stop_sweep.py — Find the optimal stop-loss for the chart
pattern strategy by sweeping stops from -3% to -10% while holding take-profit
and max-hold constant.

HOW TO RUN:
  python backtesting/chart_stop_sweep.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from backtesting._tiingo_compat import download as _yf_dl
import pandas as pd
from dataclasses import dataclass
from loguru import logger
from agents.chart_pattern_screener import detect_chart_patterns
from backtesting.monte_carlo import run as monte_carlo_run

logger.remove()
logger.add(sys.stderr, level="WARNING")

LOOKBACK_PERIOD = "2y"
WINDOW_DAYS     = 90
HOLD_DAYS_MAX   = 20
TAKE_PROFIT_PCT = 0.12   # fixed
STOP_LEVELS     = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]


@dataclass
class Trade:
    pnl_pct: float
    exit_reason: str


@dataclass
class StopStats:
    stop:       float
    trades:     int
    win_pct:    float
    avg_return: float
    avg_win:    float
    avg_loss:   float
    rr:         float
    expectancy: float
    worst:      float
    mc_median:  float
    mc_p5:      float
    mc_ruin:    float


def _load(ticker: str) -> pd.DataFrame:
    df = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                progress=False, auto_adjust=True, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def _signals_for_ticker(ticker: str, df: pd.DataFrame) -> list[tuple[int, float]]:
    """Return (entry_bar_idx, entry_price) for every confirmed BUY breakout."""
    entries = []
    for i in range(WINDOW_DAYS, len(df) - 1, 5):
        window = df.iloc[i - WINDOW_DAYS: i + 1].copy()
        try:
            signals = detect_chart_patterns(ticker, window)
        except Exception:
            continue
        buy_signals = [s for s in signals if s.state == "confirmed" and s.action == "BUY"]
        if buy_signals and i + 1 < len(df):
            entry_p = float(df.iloc[i + 1]["open"])
            entries.append((i + 1, entry_p))
    return entries


def _trades_at_stop(df: pd.DataFrame, entries: list[tuple[int, float]],
                    stop_pct: float) -> list[Trade]:
    """Replay all entries with a given stop, return trade results."""
    trades = []
    used_bars: set[int] = set()

    for entry_idx, entry_p in entries:
        # Skip if this bar was already consumed by a previous trade
        if any(entry_idx <= b <= entry_idx + HOLD_DAYS_MAX for b in used_bars):
            continue

        for j in range(entry_idx + 1, min(entry_idx + HOLD_DAYS_MAX + 1, len(df))):
            price = float(df.iloc[j]["close"])
            dd    = (price - entry_p) / entry_p
            held  = j - entry_idx
            reason = None

            if dd >= TAKE_PROFIT_PCT:   reason = "take-profit"
            elif dd <= -stop_pct:       reason = "stop-loss"
            elif held >= HOLD_DAYS_MAX: reason = "max-hold"

            if reason:
                trades.append(Trade(pnl_pct=(price - entry_p) / entry_p, exit_reason=reason))
                used_bars.update(range(entry_idx, j + 1))
                break

    return trades


def _stats(stop: float, trades: list[Trade]) -> StopStats:
    returns = [t.pnl_pct for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    win_pct = len(wins) / len(returns) if returns else 0.0
    avg_win = float(np.mean(wins))   if wins   else 0.0
    avg_loss= float(np.mean(losses)) if losses else 0.0
    rr      = abs(avg_win / avg_loss) if avg_loss else float("inf")
    exp     = win_pct * avg_win + (1 - win_pct) * avg_loss

    mc = monte_carlo_run(returns, starting_equity=100, n_runs=1000)

    return StopStats(
        stop       = stop,
        trades     = len(returns),
        win_pct    = win_pct,
        avg_return = float(np.mean(returns)),
        avg_win    = avg_win,
        avg_loss   = avg_loss,
        rr         = rr,
        expectancy = exp,
        worst      = min(returns),
        mc_median  = mc.median_return,
        mc_p5      = mc.percentile_5,
        mc_ruin    = mc.prob_ruin,
    )


def run_sweep(tickers: list[str]) -> None:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║      CHART PATTERN  STOP-LOSS  SWEEP                     ║
║  Period: 2y  |  Tickers: {len(tickers):3d}  |  Target: +{TAKE_PROFIT_PCT:.0%}          ║
║  Stops tested: {[f'-{s:.0%}' for s in STOP_LEVELS]}    ║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Step 1: download data + detect all entry signals once ─────────────────
    print("  Downloading data and detecting chart pattern signals…")
    t0 = time.time()
    ticker_data: dict[str, tuple[pd.DataFrame, list]] = {}
    for ticker in tickers:
        try:
            df = _load(ticker)
            if len(df) < WINDOW_DAYS + HOLD_DAYS_MAX:
                continue
            entries = _signals_for_ticker(ticker, df)
            if entries:
                ticker_data[ticker] = (df, entries)
        except Exception:
            continue
    total_entries = sum(len(e) for _, e in ticker_data.values())
    print(f"  Done in {time.time()-t0:.0f}s  |  {len(ticker_data)} tickers  |  {total_entries} entry signals\n")

    if not ticker_data:
        print("  No signals found — check tickers / data.")
        return

    # ── Step 2: replay each stop level against cached signals ─────────────────
    results: list[StopStats] = []

    for stop in STOP_LEVELS:
        print(f"  Testing stop = -{stop:.0%}…", end=" ", flush=True)
        t0 = time.time()
        all_trades: list[Trade] = []
        for ticker, (df, entries) in ticker_data.items():
            all_trades.extend(_trades_at_stop(df, entries, stop))

        if not all_trades:
            print("no trades.")
            continue

        st = _stats(stop, all_trades)
        results.append(st)
        print(f"{st.trades} trades  ({time.time()-t0:.0f}s)")

    if not results:
        return

    # ── Step 3: summary table ─────────────────────────────────────────────────
    print(f"""
{'─'*82}
{'Stop':>6}  {'Trades':>6}  {'Win%':>5}  {'AvgRet':>7}  {'Expect':>7}  {'R:R':>5}  {'MC Med':>7}  {'MC p5':>7}  {'Ruin':>5}
{'─'*82}""")

    for s in results:
        marker = " ◄" if s == max(results, key=lambda x: x.expectancy) else ""
        print(
            f"  -{s.stop:.0%}   {s.trades:>6}  {s.win_pct:>4.0%}  "
            f"{s.avg_return:>+6.1%}  {s.expectancy:>+6.2%}  {s.rr:>4.2f}  "
            f"{s.mc_median:>+6.0%}  {s.mc_p5:>+6.0%}  {s.mc_ruin:>4.1%}{marker}"
        )

    print(f"{'─'*82}")

    best_exp = max(results, key=lambda s: s.expectancy)
    best_mc  = max(results, key=lambda s: s.mc_median)
    best_ruin= min(results, key=lambda s: s.mc_ruin)

    print(f"\n  ✓ Best expectancy  : stop = -{best_exp.stop:.0%}  "
          f"({best_exp.expectancy:+.2%}/trade,  win {best_exp.win_pct:.0%},  ruin {best_exp.mc_ruin:.1%})")
    if best_mc.stop != best_exp.stop:
        print(f"  ✓ Best MC median   : stop = -{best_mc.stop:.0%}  "
              f"(median {best_mc.mc_median:+.0%},  ruin {best_mc.mc_ruin:.1%})")
    if best_ruin.stop != best_exp.stop:
        print(f"  ✓ Lowest ruin      : stop = -{best_ruin.stop:.0%}  "
              f"(ruin {best_ruin.mc_ruin:.1%},  expectancy {best_ruin.expectancy:+.2%})")
    print()


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    tickers = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=30) + get_dow_watchlist()
    ))
    run_sweep(tickers)
