"""
backtesting/chart_pattern_backtest.py — Backtest chart pattern breakout signals.

STRATEGY:
  Uses a rolling 90-day window to detect chart patterns at each bar.
  Entry: confirmed breakout day close
  Exit:  Take-profit at +12%  OR  Stop-loss at -6%  OR  max 20-day hold
         (chart patterns have longer expected moves than candlesticks)

Only "confirmed" BUY breakouts are traded (same as live bot).

HOW TO RUN:
  python backtesting/chart_pattern_backtest.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass
from loguru import logger
from agents.chart_pattern_screener import detect_chart_patterns
from backtesting.monte_carlo import run as monte_carlo_run, print_summary

LOOKBACK_PERIOD   = "2y"
WINDOW_DAYS       = 90    # rolling detection window
HOLD_DAYS_MAX     = 20
STOP_LOSS_PCT     = 0.10  # optimised: sweep showed -10% beats -6% on both expectancy and ruin
TAKE_PROFIT_PCT   = 0.12
POSITION_SIZE_USD = 100

# ── Liquidity filter ──────────────────────────────────────────────────────────
# Skip tickers whose 20-day average daily value (close × volume) is below this.
# Illiquid micro-caps gap through stops and produce the worst tail losses.
MIN_ADV_USD = 5_000_000   # $5M avg daily value


@dataclass
class Trade:
    ticker: str; pattern: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float; pnl_pct: float
    hold_days: int; exit_reason: str


def _load(ticker: str) -> pd.DataFrame:
    df = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                progress=False, auto_adjust=True, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def _backtest_ticker(ticker: str) -> list[Trade]:
    df = _load(ticker)
    if df is None or len(df) < WINDOW_DAYS + HOLD_DAYS_MAX:
        return []

    # ── Liquidity gate ────────────────────────────────────────────────────────
    # Compute 20-day average daily value (close × volume).  Illiquid tickers
    # gap through stops and produce the worst tail losses in the trade log.
    adv_usd = float((df["close"] * df["volume"]).rolling(20).mean().iloc[-1])
    if adv_usd < MIN_ADV_USD:
        logger.debug(f"{ticker}: ADV ${adv_usd:,.0f} < ${MIN_ADV_USD:,.0f} — skipped (illiquid)")
        return []

    trades    = []
    in_pos    = False
    entry_p   = 0.0
    entry_d   = ""
    entry_idx = 0
    entry_pat = ""

    # Scan with step=5 to reduce compute (chart patterns don't change day-to-day)
    for i in range(WINDOW_DAYS, len(df) - 1, 5):
        window = df.iloc[i - WINDOW_DAYS: i + 1].copy()

        if not in_pos:
            try:
                signals = detect_chart_patterns(ticker, window)
            except Exception:
                continue
            buy_signals = [s for s in signals if s.state == "confirmed" and s.action == "BUY"]
            if not buy_signals:
                continue
            if i + 1 >= len(df):
                continue
            entry_p   = float(df.iloc[i + 1]["open"])
            entry_d   = str(df.index[i + 1].date())
            entry_idx = i + 1
            entry_pat = buy_signals[0].pattern
            in_pos    = True
        else:
            # Check each day until exit condition
            for j in range(entry_idx + 1, min(entry_idx + HOLD_DAYS_MAX + 1, len(df))):
                price = float(df.iloc[j]["close"])
                dd    = (price - entry_p) / entry_p
                held  = j - entry_idx
                reason = None

                if dd >= TAKE_PROFIT_PCT:   reason = f"Take-profit ({dd:+.1%})"
                elif dd <= -STOP_LOSS_PCT:  reason = f"Stop-loss ({dd:+.1%})"
                elif held >= HOLD_DAYS_MAX: reason = f"Max hold ({HOLD_DAYS_MAX}d)"

                if reason:
                    trades.append(Trade(ticker, entry_pat, entry_d, str(df.index[j].date()),
                        entry_p, price, (price - entry_p) / entry_p, held, reason))
                    in_pos = False
                    i = j  # advance outer scan
                    break

    return trades


def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No chart pattern trades generated."); return

    returns  = [t.pnl_pct for t in all_trades]
    wins     = [t for t in all_trades if t.pnl_pct > 0]
    losses   = [t for t in all_trades if t.pnl_pct <= 0]
    avg_win  = np.mean([t.pnl_pct for t in wins])  if wins   else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss else float("inf")

    pat_stats: dict[str, list[float]] = {}
    for t in all_trades:
        pat_stats.setdefault(t.pattern, []).append(t.pnl_pct)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Chart Pattern Backtest  (confirmed BUY only)
 Stop: -{STOP_LOSS_PCT:.0%}  |  Target: +{TAKE_PROFIT_PCT:.0%}  |  Max hold: {HOLD_DAYS_MAX}d
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Tickers tested : {len(set(t.ticker for t in all_trades))}
 Total trades   : {len(all_trades)}
 Winners        : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers         : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win        : {avg_win:+.1%}
 Avg loss       : {avg_loss:+.1%}
 Risk/Reward    : {rr:.2f}:1
 Best trade     : {max(returns):+.1%}
 Worst trade    : {min(returns):+.1%}
 Avg return     : {np.mean(returns):+.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pattern breakdown:
""")
    for pat, rets in sorted(pat_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
        w = sum(1 for r in rets if r > 0)
        print(f"  {pat:28s}  n={len(rets):3d}  win={w/len(rets):.0%}  avg={np.mean(rets):+.1%}")

    print("""
 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(f"  {t.ticker:6s}  {t.pattern:25s}  {t.entry_date} → {t.exit_date}  "
              f"hold={t.hold_days:2d}d  {t.pnl_pct:+6.1%}  ({t.exit_reason})")


def run_backtest(tickers: list[str]) -> list[float]:
    all_trades: list[Trade] = []
    for ticker in tickers:
        logger.info(f"Chart pattern: {ticker}…")
        trades = _backtest_ticker(ticker)
        if trades:
            logger.info(f"  {ticker}: {len(trades)} trades")
        all_trades.extend(trades)
    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    print(f"\nRunning Chart Pattern Backtest on {len(CUSTOM_WATCHLIST)} tickers…")
    returns = run_backtest(CUSTOM_WATCHLIST)
    if returns:
        print("\nRunning Monte Carlo…")
        result = monte_carlo_run(returns, starting_equity=POSITION_SIZE_USD, n_runs=1000)
        print_summary(result)
