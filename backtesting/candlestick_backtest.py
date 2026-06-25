"""
backtesting/candlestick_backtest.py — Backtest candlestick pattern signals.

STRATEGY:
  Entry: Enter at open of next bar after a strong BUY pattern fires
  Exit:  Take-profit +8%  |  Stop-loss -5%  |  Max 10-day hold

PERFORMANCE NOTE:
  The vol-compression check uses a 252-bar percentile rank (O(n^2) rolling apply).
  To avoid recomputing it at every bar, we pre-compute ATR rank + BB rank once per
  ticker, then only call detect_patterns() on candidate bars that pass the cheap
  pre-filter. This keeps runtime manageable across 50+ tickers.

  SPY is downloaded once and reused across all tickers for the RS / regime checks.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass
from loguru import logger
from agents.candlestick_patterns import detect_patterns, _atr, _pct_rank
from backtesting.monte_carlo import run as monte_carlo_run, print_summary

LOOKBACK_PERIOD   = "2y"
MIN_HISTORY_BARS  = 252    # need a full year for percentile ranks to be valid
HOLD_DAYS_MAX     = 10
STOP_LOSS_PCT     = 0.05
TAKE_PROFIT_PCT   = 0.08
POSITION_SIZE_USD = 100
ATR_PRE_FILTER    = 45.0   # skip bars where ATR rank clearly too high
BB_PRE_FILTER     = 45.0   # skip bars where BB rank clearly too high


@dataclass
class Trade:
    ticker: str
    pattern: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    hold_days: int
    exit_reason: str


def _load(ticker: str) -> pd.DataFrame:
    df = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                progress=False, auto_adjust=True, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def _load_spy() -> pd.DataFrame:
    return _load("SPY")


def _backtest_ticker(ticker: str, spy_df: pd.DataFrame) -> list[Trade]:
    df = _load(ticker)
    if df is None or len(df) < MIN_HISTORY_BARS + HOLD_DAYS_MAX:
        return []

    # Pre-compute expensive rolling series once for the whole ticker.
    # Avoids recomputing O(n^2) percentile ranks at every bar.
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    atr_s    = _atr(high, low, close, 14)
    atr_rank = _pct_rank(atr_s, window=252, min_periods=60)

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_width = (2 * std20) / sma20.replace(0, float("nan")) * 100
    bb_rank  = _pct_rank(bb_width, window=252, min_periods=60)

    trades    = []
    in_pos    = False
    entry_p   = 0.0
    entry_d   = ""
    entry_idx = 0
    entry_pat = ""

    for i in range(MIN_HISTORY_BARS, len(df) - 1):
        if not in_pos:
            # Cheap pre-filter: skip if ATR/BB ranks clearly too high
            ar = atr_rank.iloc[i]
            br = bb_rank.iloc[i]
            if pd.isna(ar) or pd.isna(br):
                continue
            if float(ar) > ATR_PRE_FILTER or float(br) > BB_PRE_FILTER:
                continue

            # Align SPY history to match ticker dates up to bar i
            ticker_slice = df.iloc[: i + 1].copy()
            spy_slice = spy_df.reindex(ticker_slice.index).ffill().dropna()

            # Full signal check only on candidate bars
            try:
                signals = detect_patterns(
                    ticker,
                    ticker_slice,
                    spy_df=spy_slice,
                )
            except Exception:
                continue
            buy_sigs = [s for s in signals if s.action == "BUY" and s.strength == "strong"]
            if not buy_sigs:
                continue
            if i + 1 >= len(df):
                continue
            entry_p   = float(df.iloc[i + 1]["open"])
            entry_d   = str(df.index[i + 1].date())
            entry_idx = i + 1
            entry_pat = buy_sigs[0].pattern
            in_pos    = True
        else:
            price = float(df.iloc[i]["close"])
            dd    = (price - entry_p) / entry_p
            held  = i - entry_idx
            reason = None
            if dd >= TAKE_PROFIT_PCT:
                reason = f"Take-profit ({dd:+.1%})"
            elif dd <= -STOP_LOSS_PCT:
                reason = f"Stop-loss ({dd:+.1%})"
            elif held >= HOLD_DAYS_MAX:
                reason = f"Max hold ({HOLD_DAYS_MAX}d)"
            if reason:
                trades.append(Trade(
                    ticker, entry_pat, entry_d,
                    str(df.index[i].date()), entry_p, price,
                    (price - entry_p) / entry_p, held, reason,
                ))
                in_pos = False

    return trades


def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No candlestick trades generated.")
        return

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
 Candlestick Pattern Backtest  (ATR<={ATR_PRE_FILTER}pct / BB<={BB_PRE_FILTER}pct pre-filter)
 Stop: -{STOP_LOSS_PCT:.0%}  |  Target: +{TAKE_PROFIT_PCT:.0%}  |  Max: {HOLD_DAYS_MAX}d
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
""")
    print(" Pattern breakdown:\n")
    for pat, rets in sorted(pat_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
        w = sum(1 for r in rets if r > 0)
        pct = w / len(rets)
        avg = np.mean(rets)
        print(f"  {pat:35s}  n={len(rets):3d}  win={pct:.0%}  avg={avg:+.1%}")

    print("\n Trade log (best to worst):\n")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.ticker:6s}  {t.pattern:30s}  "
            f"{t.entry_date} -> {t.exit_date}  "
            f"hold={t.hold_days:2d}d  {t.pnl_pct:+6.1%}  ({t.exit_reason})"
        )


def run_backtest(tickers: list[str]) -> list[float]:
    logger.info("Loading SPY for RS / regime checks...")
    spy_df = _load_spy()

    all_trades: list[Trade] = []
    for ticker in tickers:
        logger.info(f"Candlestick: {ticker}...")
        trades = _backtest_ticker(ticker, spy_df)
        if trades:
            logger.info(f"  {ticker}: {len(trades)} trades")
        all_trades.extend(trades)
    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    print(f"\nRunning Candlestick Backtest on {len(CUSTOM_WATCHLIST)} tickers...")
    returns = run_backtest(CUSTOM_WATCHLIST)
    if returns:
        result = monte_carlo_run(returns, starting_equity=POSITION_SIZE_USD, n_runs=1000)
        print_summary(result)
