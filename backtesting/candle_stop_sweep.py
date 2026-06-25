"""
backtesting/candle_stop_sweep.py — Find the optimal stop-loss AND take-profit
for the candlestick strategy by sweeping stops from -2% to -8% crossed with
take-profits from +5% to +15%.

HOW TO RUN:
  python backtesting/candle_stop_sweep.py

Signals are detected once per ticker; all stop/target combos replay against
the same cached entry list — so the full grid runs in roughly the same time
as a single backtest.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from backtesting._tiingo_compat import download as _yf_dl
import pandas as pd
from dataclasses import dataclass
from loguru import logger
from agents.candlestick_patterns import detect_patterns
from backtesting.monte_carlo import run as monte_carlo_run

logger.remove()
logger.add(sys.stderr, level="WARNING")

LOOKBACK_PERIOD = "2y"
WINDOW_BARS     = 20
HOLD_DAYS_MAX   = 10

STOP_LEVELS        = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
TAKE_PROFIT_LEVELS = [0.05, 0.08, 0.10, 0.12, 0.15]


@dataclass
class Trade:
    pnl_pct: float
    exit_reason: str


@dataclass
class GridStats:
    stop:       float
    target:     float
    trades:     int
    win_pct:    float
    avg_return: float
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


def _entries_for_ticker(ticker: str, df: pd.DataFrame) -> list[tuple[int, float, str]]:
    """Return (entry_bar_idx, entry_price, pattern) for every strong BUY signal."""
    entries = []
    for i in range(WINDOW_BARS, len(df) - 1):
        window = df.iloc[i - WINDOW_BARS: i + 1].copy()
        if len(window) < 15:
            continue
        try:
            signals = detect_patterns(ticker, window)
        except Exception:
            continue
        buy_sigs = [s for s in signals if s.action == "BUY" and s.strength == "strong"]
        if buy_sigs and i + 1 < len(df):
            entry_p = float(df.iloc[i + 1]["open"])
            entries.append((i + 1, entry_p, buy_sigs[0].pattern))
    return entries


def _trades_at_params(df: pd.DataFrame,
                      entries: list[tuple[int, float, str]],
                      stop_pct: float,
                      target_pct: float) -> list[Trade]:
    trades = []
    used: set[int] = set()

    for entry_idx, entry_p, pattern in entries:
        if any(entry_idx <= b <= entry_idx + HOLD_DAYS_MAX for b in used):
            continue
        for j in range(entry_idx + 1, min(entry_idx + HOLD_DAYS_MAX + 1, len(df))):
            price  = float(df.iloc[j]["close"])
            dd     = (price - entry_p) / entry_p
            held   = j - entry_idx
            reason = None
            if dd >= target_pct:        reason = "take-profit"
            elif dd <= -stop_pct:       reason = "stop-loss"
            elif held >= HOLD_DAYS_MAX: reason = "max-hold"
            if reason:
                trades.append(Trade(pnl_pct=(price - entry_p) / entry_p,
                                    exit_reason=reason))
                used.update(range(entry_idx, j + 1))
                break
    return trades


def _stats(stop: float, target: float, trades: list[Trade]) -> GridStats:
    returns = [t.pnl_pct for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    win_pct = len(wins) / len(returns) if returns else 0.0
    avg_win = float(np.mean(wins))   if wins   else 0.0
    avg_loss= float(np.mean(losses)) if losses else 0.0
    rr      = abs(avg_win / avg_loss) if avg_loss else float("inf")
    exp     = win_pct * avg_win + (1 - win_pct) * avg_loss
    mc = monte_carlo_run(returns, starting_equity=100, n_runs=1000)
    return GridStats(
        stop=stop, target=target, trades=len(returns),
        win_pct=win_pct, avg_return=float(np.mean(returns)),
        rr=rr, expectancy=exp, worst=min(returns),
        mc_median=mc.median_return, mc_p5=mc.percentile_5,
        mc_ruin=mc.prob_ruin,
    )


def run_sweep(tickers: list[str]) -> None:
    n_combos = len(STOP_LEVELS) * len(TAKE_PROFIT_LEVELS)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║      CANDLESTICK  STOP × TARGET  SWEEP                   ║
║  Period: 2y  |  Tickers: {len(tickers):3d}  |  Max hold: {HOLD_DAYS_MAX}d          ║
║  Stops: {[f'-{s:.0%}' for s in STOP_LEVELS]}            ║
║  Targets: {[f'+{t:.0%}' for t in TAKE_PROFIT_LEVELS]}          ║
║  Grid: {n_combos} combinations                                    ║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Step 1: download + detect signals once ────────────────────────────────
    print("  Downloading data and detecting candlestick signals…")
    t0 = time.time()
    ticker_data: dict[str, tuple[pd.DataFrame, list]] = {}
    for ticker in tickers:
        try:
            df = _load(ticker)
            if len(df) < WINDOW_BARS + HOLD_DAYS_MAX:
                continue
            entries = _entries_for_ticker(ticker, df)
            if entries:
                ticker_data[ticker] = (df, entries)
        except Exception:
            continue
    total_entries = sum(len(e) for _, e in ticker_data.values())
    print(f"  Done in {time.time()-t0:.0f}s  |  {len(ticker_data)} tickers  |  {total_entries} signals\n")

    if not ticker_data:
        print("  No signals found.")
        return

    # ── Step 2: grid search ───────────────────────────────────────────────────
    results: list[GridStats] = []

    for stop in STOP_LEVELS:
        row_results = []
        for target in TAKE_PROFIT_LEVELS:
            all_trades: list[Trade] = []
            for ticker, (df, entries) in ticker_data.items():
                all_trades.extend(_trades_at_params(df, entries, stop, target))
            if all_trades:
                row_results.append(_stats(stop, target, all_trades))
        results.extend(row_results)

    if not results:
        return

    # ── Step 3: print grid table (stop rows × target cols) ───────────────────
    print("  Expectancy grid  (stop rows × target cols)\n")
    header = f"  {'Stop':>5}  " + "  ".join(f"+{t:.0%} Exp / Ruin" for t in TAKE_PROFIT_LEVELS)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for stop in STOP_LEVELS:
        row_vals = []
        for target in TAKE_PROFIT_LEVELS:
            match = next((r for r in results if r.stop == stop and r.target == target), None)
            if match:
                row_vals.append(f"{match.expectancy:>+5.2%}/{match.mc_ruin:>4.1%}")
            else:
                row_vals.append("   —  /  — ")
        print(f"  -{stop:.0%}    " + "    ".join(row_vals))

    # ── Step 4: full ranked table ─────────────────────────────────────────────
    print(f"""
{'─'*86}
{'Stop':>5} {'Target':>7}  {'Trades':>6}  {'Win%':>5}  {'Expect':>7}  {'R:R':>5}  {'MC Med':>7}  {'MC p5':>7}  {'Ruin':>5}
{'─'*86}""")

    best_exp = max(results, key=lambda r: r.expectancy)
    for r in sorted(results, key=lambda r: r.expectancy, reverse=True)[:15]:
        marker = " ◄" if r is best_exp else ""
        print(
            f"  -{r.stop:.0%}  +{r.target:.0%}   {r.trades:>6}  {r.win_pct:>4.0%}  "
            f"{r.expectancy:>+6.2%}  {r.rr:>4.2f}  "
            f"{r.mc_median:>+6.0%}  {r.mc_p5:>+6.0%}  {r.mc_ruin:>4.1%}{marker}"
        )
    print(f"{'─'*86}")

    best_mc   = max(results, key=lambda r: r.mc_median)
    best_ruin = min(results, key=lambda r: r.mc_ruin)

    print(f"\n  ✓ Best expectancy : stop=-{best_exp.stop:.0%}  target=+{best_exp.target:.0%}"
          f"  ({best_exp.expectancy:+.2%}/trade,  win {best_exp.win_pct:.0%},  ruin {best_exp.mc_ruin:.1%})")
    if best_mc is not best_exp:
        print(f"  ✓ Best MC median  : stop=-{best_mc.stop:.0%}  target=+{best_mc.target:.0%}"
              f"  (median {best_mc.mc_median:+.0%},  ruin {best_mc.mc_ruin:.1%})")
    if best_ruin is not best_exp:
        print(f"  ✓ Lowest ruin     : stop=-{best_ruin.stop:.0%}  target=+{best_ruin.target:.0%}"
              f"  (ruin {best_ruin.mc_ruin:.1%},  expectancy {best_ruin.expectancy:+.2%})")
    print()


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    tickers = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=30) + get_dow_watchlist()
    ))
    run_sweep(tickers)
