"""
backtesting/strategy_backtest.py — RSI + Golden Cross strategy backtester.

STRATEGY RULES:
  BUY  when: RSI < 45  AND  50-day MA crosses above 200-day MA (golden cross)
  SELL when: RSI > 70  OR   50-day MA crosses below 200-day MA (death cross)

HOW TO RUN:
  python backtesting/strategy_backtest.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass
from loguru import logger
from backtesting.monte_carlo import run as monte_carlo_run, print_summary


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "NVDA", "MSFT", "META", "TSLA", "AMZN", "GOOGL"]
LOOKBACK_PERIOD = "2y"       # how far back to test
RSI_BUY_THRESHOLD  = 45      # buy when RSI below this
RSI_SELL_THRESHOLD = 70      # sell when RSI above this
POSITION_SIZE_USD  = 1000    # dollars per trade


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str


# ─── Core backtest logic ──────────────────────────────────────────────────────

def _load_data(ticker: str) -> pd.DataFrame:
    df = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Moving averages
    df["ma50"]  = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    # Golden cross / death cross flags
    df["golden_cross"] = (df["ma50"] > df["ma200"]) & (df["ma50"].shift(1) <= df["ma200"].shift(1))
    df["death_cross"]  = (df["ma50"] < df["ma200"]) & (df["ma50"].shift(1) >= df["ma200"].shift(1))

    return df.dropna()


def _backtest_ticker(ticker: str) -> list[Trade]:
    """Simulate trades on one ticker and return the trade log."""
    df = _load_data(ticker)
    if df.empty or len(df) < 210:
        logger.warning(f"{ticker}: not enough data, skipping.")
        return []

    df = _compute_signals(df)
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date  = ""

    for i in range(1, len(df)):
        row  = df.iloc[i]
        date = str(df.index[i].date())

        if not in_position:
            # BUY signal: golden cross AND RSI not overbought
            if row["golden_cross"] and row["rsi"] < RSI_BUY_THRESHOLD:
                in_position  = True
                entry_price  = float(row["close"])
                entry_date   = date
                logger.debug(f"{ticker} BUY  {date}  ${entry_price:.2f}  RSI={row['rsi']:.1f}")

        else:
            # SELL signal: RSI overbought OR death cross
            exit_reason = None
            if row["rsi"] > RSI_SELL_THRESHOLD:
                exit_reason = "RSI overbought"
            elif row["death_cross"]:
                exit_reason = "Death cross"

            if exit_reason:
                exit_price = float(row["close"])
                pnl_pct    = (exit_price - entry_price) / entry_price
                trades.append(Trade(
                    ticker      = ticker,
                    entry_date  = entry_date,
                    exit_date   = date,
                    entry_price = entry_price,
                    exit_price  = exit_price,
                    pnl_pct     = pnl_pct,
                    exit_reason = exit_reason,
                ))
                logger.debug(
                    f"{ticker} SELL {date}  ${exit_price:.2f}  "
                    f"P&L={pnl_pct:+.1%}  ({exit_reason})"
                )
                in_position = False

    return trades


# ─── Report ───────────────────────────────────────────────────────────────────

def _print_trade_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No trades were generated.")
        return

    wins   = [t for t in all_trades if t.pnl_pct > 0]
    losses = [t for t in all_trades if t.pnl_pct <= 0]
    returns = [t.pnl_pct for t in all_trades]

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RSI + Golden Cross  —  Trade Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total trades   : {len(all_trades)}
 Winners        : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers         : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win        : {np.mean([t.pnl_pct for t in wins]):+.1%}
 Avg loss       : {np.mean([t.pnl_pct for t in losses]):+.1%}
 Best trade     : {max(returns):+.1%}
 Worst trade    : {min(returns):+.1%}
 Avg return     : {np.mean(returns):+.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Individual trades:
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.ticker:6s}  {t.entry_date} → {t.exit_date}  "
            f"{t.pnl_pct:+6.1%}  ({t.exit_reason})"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(tickers: list[str] = DEFAULT_TICKERS) -> list[float]:
    """Run the full backtest and return list of trade returns for Monte Carlo."""
    all_trades: list[Trade] = []

    for ticker in tickers:
        logger.info(f"Backtesting {ticker}…")
        trades = _backtest_ticker(ticker)
        logger.info(f"{ticker}: {len(trades)} trades found")
        all_trades.extend(trades)

    _print_trade_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    print("\nRunning RSI + Golden Cross backtest…")
    print(f"Tickers : {', '.join(DEFAULT_TICKERS)}")
    print(f"Period  : {LOOKBACK_PERIOD}\n")

    trade_returns = run_backtest(DEFAULT_TICKERS)

    if trade_returns:
        print("\n\nRunning Monte Carlo simulation on backtest results…")
        result = monte_carlo_run(
            trade_returns,
            starting_equity=POSITION_SIZE_USD,
            n_runs=1000,
        )
        print_summary(result)
    else:
        print("\nNo trades generated — try adjusting RSI thresholds or a longer period.")
