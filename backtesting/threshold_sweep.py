"""
backtesting/threshold_sweep.py — Find the optimal BUY score threshold for the
AI Stock Screener by running a full 2-year backtest at each cutoff from 5 to 9.

HOW TO RUN:
  python backtesting/threshold_sweep.py

Output: ranked comparison table + per-threshold Monte Carlo summary.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataclasses import dataclass
from loguru import logger
from backtesting.screener_backtest import _load_and_compute, Trade
from backtesting.monte_carlo import run as monte_carlo_run

logger.remove()
logger.add(sys.stderr, level="WARNING")   # suppress INFO noise

STOP_LOSS_PCT = 0.10
SELL_SCORE    = 5           # exit when score drops below this (fixed)
LOOKBACK      = "2y"


# ─── Single-threshold backtest ────────────────────────────────────────────────

def _backtest_at_threshold(tickers: list[str], buy_score: int) -> list[Trade]:
    all_trades: list[Trade] = []
    for ticker in tickers:
        try:
            df = _load_and_compute(ticker)
        except Exception:
            continue
        if df.empty:
            continue

        in_pos = False
        entry_price = entry_idx = entry_score = 0
        entry_date = ""

        for i in range(1, len(df)):
            row   = df.iloc[i]
            date  = str(df.index[i].date())
            score = int(row["score"])
            rsi   = float(row["rsi_14"])
            price = float(row["close"])
            stoch = str(row["stoch_signal"])

            if not in_pos:
                if score >= buy_score:
                    in_pos = True
                    entry_price = price
                    entry_date  = date
                    entry_score = score
                    entry_idx   = i
            else:
                dd = (price - entry_price) / entry_price
                reason = None
                if dd <= -STOP_LOSS_PCT:
                    reason = "Stop-loss"
                elif score < SELL_SCORE:
                    reason = "Score dropped"
                elif rsi > 75 and stoch == "overbought":
                    reason = "RSI+Stoch exit"
                elif rsi > 75:
                    reason = "RSI exit"

                if reason:
                    all_trades.append(Trade(
                        ticker, entry_date, date,
                        entry_price, price,
                        (price - entry_price) / entry_price,
                        i - entry_idx, entry_score, reason
                    ))
                    in_pos = False

    return all_trades


# ─── Stats helper ─────────────────────────────────────────────────────────────

@dataclass
class ThresholdStats:
    threshold:   int
    trades:      int
    win_pct:     float
    avg_return:  float
    avg_win:     float
    avg_loss:    float
    rr:          float          # reward/risk ratio
    expectancy:  float          # per-trade expectancy
    best:        float
    worst:       float
    mc_median:   float
    mc_p5:       float
    mc_p95:      float
    mc_ruin:     float


def _stats(threshold: int, trades: list[Trade]) -> ThresholdStats:
    returns = [t.pnl_pct for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    win_pct = len(wins) / len(returns) if returns else 0.0
    avg_win = np.mean(wins)   if wins   else 0.0
    avg_loss= np.mean(losses) if losses else 0.0
    rr      = abs(avg_win / avg_loss) if avg_loss else float("inf")
    exp     = win_pct * avg_win + (1 - win_pct) * avg_loss

    mc = monte_carlo_run(returns, starting_equity=100, n_runs=1000)

    return ThresholdStats(
        threshold  = threshold,
        trades     = len(returns),
        win_pct    = win_pct,
        avg_return = float(np.mean(returns)),
        avg_win    = avg_win,
        avg_loss   = avg_loss,
        rr         = rr,
        expectancy = exp,
        best       = max(returns),
        worst      = min(returns),
        mc_median  = mc.median_return,
        mc_p5      = mc.percentile_5,
        mc_p95     = mc.percentile_95,
        mc_ruin    = mc.prob_ruin,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_sweep(tickers: list[str], thresholds: list[int] | None = None) -> None:
    if thresholds is None:
        thresholds = [5, 6, 7, 8, 9]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║        SCREENER  BUY-THRESHOLD  SWEEP                    ║
║  Period: 2y  |  Tickers: {len(tickers):3d}  |  Stop: -{STOP_LOSS_PCT:.0%}            ║
║  Thresholds tested: {thresholds}                  ║
╚══════════════════════════════════════════════════════════╝
""")

    # Download data once per ticker, re-use across all thresholds
    print("  Downloading & computing indicators for all tickers…")
    t0 = time.time()
    dfs = {}
    for ticker in tickers:
        try:
            df = _load_and_compute(ticker)
            if not df.empty:
                dfs[ticker] = df
        except Exception:
            pass
    print(f"  Done in {time.time()-t0:.0f}s  |  {len(dfs)} tickers with data\n")

    results: list[ThresholdStats] = []

    for thresh in thresholds:
        print(f"  Running threshold = {thresh}…", end=" ", flush=True)
        t0 = time.time()

        # Replay trades from cached DFs
        trades = _backtest_at_threshold(list(dfs.keys()), buy_score=thresh)
        # Patch: re-run against cached dfs directly
        all_trades: list[Trade] = []
        for ticker, df in dfs.items():
            in_pos = False
            entry_price = entry_idx = entry_score = 0
            entry_date = ""
            for i in range(1, len(df)):
                row   = df.iloc[i]
                date  = str(df.index[i].date())
                score = int(row["score"])
                rsi   = float(row["rsi_14"])
                price = float(row["close"])
                stoch = str(row["stoch_signal"])
                if not in_pos:
                    if score >= thresh:
                        in_pos = True; entry_price = price
                        entry_date = date; entry_score = score; entry_idx = i
                else:
                    dd = (price - entry_price) / entry_price
                    reason = None
                    if dd <= -STOP_LOSS_PCT:                  reason = "Stop-loss"
                    elif score < SELL_SCORE:                  reason = "Score dropped"
                    elif rsi > 75 and stoch == "overbought":  reason = "RSI+Stoch exit"
                    elif rsi > 75:                            reason = "RSI exit"
                    if reason:
                        all_trades.append(Trade(ticker, entry_date, date,
                            entry_price, price,
                            (price - entry_price) / entry_price,
                            i - entry_idx, entry_score, reason))
                        in_pos = False

        if not all_trades:
            print(f"no trades at threshold {thresh}, skipping.")
            continue

        st = _stats(thresh, all_trades)
        results.append(st)
        print(f"{st.trades} trades  ({time.time()-t0:.0f}s)")

    if not results:
        print("No results — check tickers / data.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"""
{'─'*85}
{'Threshold':>9}  {'Trades':>6}  {'Win%':>5}  {'AvgRet':>7}  {'Expect':>7}  {'R:R':>5}  {'MC Med':>7}  {'MC p5':>7}  {'Ruin':>5}
{'─'*85}""")

    for s in results:
        print(
            f"  score>={s.threshold}  {s.trades:>6}  {s.win_pct:>4.0%}  "
            f"{s.avg_return:>+6.1%}  {s.expectancy:>+6.2%}  {s.rr:>4.2f}  "
            f"{s.mc_median:>+6.0%}  {s.mc_p5:>+6.0%}  {s.mc_ruin:>4.1%}"
        )

    print(f"{'─'*85}")

    # Rank by expectancy (most reliable single metric)
    best = max(results, key=lambda s: s.expectancy)
    print(f"\n  ✓ Best by expectancy : score >= {best.threshold}  "
          f"({best.expectancy:+.2%}/trade,  {best.win_pct:.0%} win,  {best.trades} trades)\n")

    # Also highlight best Monte Carlo median
    best_mc = max(results, key=lambda s: s.mc_median)
    if best_mc.threshold != best.threshold:
        print(f"  ✓ Best Monte Carlo median : score >= {best_mc.threshold}  "
              f"(median {best_mc.mc_median:+.0%},  ruin {best_mc.mc_ruin:.1%})\n")


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    tickers = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=30) + get_dow_watchlist()
    ))
    run_sweep(tickers)
