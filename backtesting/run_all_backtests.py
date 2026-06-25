"""
backtesting/run_all_backtests.py — Run ALL strategy backtests and produce a
unified performance report with Monte Carlo simulation.

Strategies tested:
  1. AI Stock Screener     (RSI · ADX · ATR%ile · OBV · RelVol — orthogonal)
  2. Chart Patterns        (12 patterns: Double Top/Bottom, H&S, Triangles, etc.)
  3. Crypto Mean Reversion (Lower BB + RSI<35 + volume spike)
  4. Walk-Forward Validation (IS/OOS rolling windows on screener strategy)

HOW TO RUN:
  python backtesting/run_all_backtests.py
  python backtesting/run_all_backtests.py --skip-walk-forward   (fast mode)
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force stdout/stderr to UTF-8 so the banner (box-drawing chars, em dashes,
# clock + warning symbols below) and any unicode in downstream backtest output
# never crash this script on a default Windows console (cp1252). Matches the
# PYTHONIOENCODING=utf-8 env var that start_all.ps1 sets for the live bot.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from loguru import logger
from backtesting.monte_carlo import run as monte_carlo_run, print_summary

logger.remove()
logger.add(sys.stderr, level="WARNING")   # suppress INFO noise during bulk run

CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "AVAX-USD", "LINK-USD", "DOGE-USD", "XRP-USD",
]


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_all(skip_walk_forward: bool = False) -> None:
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    stock_tickers = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=30) + get_dow_watchlist()
    ))

    print(f"""
╔══════════════════════════════════════════════════════════╗
║            FULL STRATEGY BACKTEST SUITE                  ║
║  Period: 2 years  |  Stocks: {len(stock_tickers):3d}  |  Crypto: {len(CRYPTO_TICKERS)}          ║
╚══════════════════════════════════════════════════════════╝
""")

    all_returns: dict[str, list[float]] = {}

    # ── 1. AI Stock Screener (orthogonal) ─────────────────────────────────────
    section("1 / 4  —  AI Stock Screener  (orthogonal indicators)")
    t0 = time.time()
    try:
        from backtesting.screener_backtest import run_backtest as run_screener
        logger.enable("backtesting.screener_backtest")
        returns = run_screener(stock_tickers)
        all_returns["Screener"] = returns
        print(f"\n  ⏱  {time.time()-t0:.0f}s  |  {len(returns)} trades")
    except Exception as exc:
        print(f"\n  ⚠  Screener backtest failed: {exc}")
        import traceback; traceback.print_exc()
        all_returns["Screener"] = []

    # ── 2. Chart Patterns ─────────────────────────────────────────────────────
    section("2 / 4  —  Chart Patterns")
    t0 = time.time()
    try:
        from backtesting.chart_pattern_backtest import run_backtest as run_chart
        returns = run_chart(stock_tickers)
        all_returns["Chart Pattern"] = returns
        print(f"\n  ⏱  {time.time()-t0:.0f}s  |  {len(returns)} trades")
    except Exception as exc:
        print(f"\n  ⚠  Chart pattern backtest failed: {exc}")
        all_returns["Chart Pattern"] = []

    # ── 3. Crypto Mean Reversion ──────────────────────────────────────────────
    section("3 / 4  —  Crypto Mean Reversion")
    t0 = time.time()
    try:
        from backtesting.crypto_mean_reversion import run_backtest as run_crypto
        returns = run_crypto(CRYPTO_TICKERS)
        all_returns["Crypto MeanRev"] = returns
        print(f"\n  ⏱  {time.time()-t0:.0f}s  |  {len(returns)} trades")
    except Exception as exc:
        print(f"\n  ⚠  Crypto backtest failed: {exc}")
        all_returns["Crypto MeanRev"] = []

    # ── 4. Institutional Breakout (Setup B) ──────────────────────────────────
    section("4 / 5  —  Institutional Breakout  (Setup B)")
    t0 = time.time()
    try:
        from backtesting.institutional_breakout_backtest import run_backtest as ib_run, DEFAULT_UNIVERSE
        print(f"  Universe: {len(DEFAULT_UNIVERSE)} tickers (growth + large-cap)")
        returns = ib_run()
        all_returns["Institutional Breakout"] = returns
        print(f"\n  ⏱  {time.time()-t0:.0f}s  |  {len(returns)} trades")
    except Exception as exc:
        print(f"\n  ⚠  IB backtest failed: {exc}")
        all_returns["Institutional Breakout"] = []

    # ── 5. Walk-Forward Validation ────────────────────────────────────────────
    if not skip_walk_forward:
        section("5 / 5  —  Walk-Forward Validation  (IS 12m / OOS 3m / step 3m)")
        t0 = time.time()
        try:
            from backtesting.walk_forward import run_walk_forward, print_report
            # Use a representative subset — walk-forward is slow (downloads + grid search)
            wf_tickers = list(dict.fromkeys(
                CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=15)
            ))
            print(f"  Tickers: {len(wf_tickers)}  (IS=12m  OOS=3m  step=3m)")
            result = run_walk_forward(wf_tickers, verbose=False)
            print_report(result)
            oos_rets = [t["pnl_pct"] for t in result.get("oos_trades", [])]
            all_returns["Walk-Forward OOS"] = oos_rets
            print(f"\n  ⏱  {time.time()-t0:.0f}s  |  {len(oos_rets)} OOS trades")
        except Exception as exc:
            print(f"\n  ⚠  Walk-forward failed: {exc}")
            import traceback; traceback.print_exc()
            all_returns["Walk-Forward OOS"] = []
    else:
        print("\n  [Walk-forward skipped — use without --skip-walk-forward to include]\n")
        all_returns["Walk-Forward OOS"] = []

    # ── Combined summary ───────────────────────────────────────────────────────
    section("COMBINED STRATEGY SUMMARY")

    total_trades = sum(len(v) for v in all_returns.values())
    all_combined: list[float] = []
    for rets in all_returns.values():
        all_combined.extend(rets)

    print(f"{'Strategy':<22} {'Trades':>6}  {'Win%':>5}  {'AvgRet':>7}  {'Best':>7}  {'Worst':>7}")
    print("-" * 65)
    for strat, rets in all_returns.items():
        if not rets:
            print(f"  {strat:<20}  {'—':>6}  {'—':>5}  {'—':>7}  {'—':>7}  {'—':>7}")
            continue
        wins    = [r for r in rets if r > 0]
        win_pct = len(wins) / len(rets)
        print(f"  {strat:<20}  {len(rets):>6}  {win_pct:>4.0%}  "
              f"  {np.mean(rets):>+6.1%}  {max(rets):>+6.1%}  {min(rets):>+6.1%}")

    print("-" * 65)
    if all_combined:
        wins    = [r for r in all_combined if r > 0]
        win_pct = len(wins) / len(all_combined)
        print(f"  {'TOTAL':<20}  {total_trades:>6}  {win_pct:>4.0%}  "
              f"  {np.mean(all_combined):>+6.1%}  {max(all_combined):>+6.1%}  {min(all_combined):>+6.1%}")

    # ── Monte Carlo on combined trade log ─────────────────────────────────────
    if all_combined:
        print("\n\n  Monte Carlo — 1,000 paths on combined trade log ($100/trade)\n")
        result = monte_carlo_run(all_combined, starting_equity=100, n_runs=1000)
        print_summary(result)

        # Per-strategy Monte Carlo
        print("\n  Per-strategy Monte Carlo ($100/trade):\n")
        for strat, rets in all_returns.items():
            if len(rets) < 5:
                continue
            r = monte_carlo_run(rets, starting_equity=100, n_runs=1000)
            print(f"  {strat:<22}  median={r.median_return:+.1%}  "
                  f"5th={r.percentile_5:+.1%}  95th={r.percentile_95:+.1%}  "
                  f"ruin={r.prob_ruin:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full strategy backtest suite")
    parser.add_argument(
        "--skip-walk-forward", action="store_true",
        help="Skip the walk-forward validation (much faster)"
    )
    args = parser.parse_args()
    run_all(skip_walk_forward=args.skip_walk_forward)
