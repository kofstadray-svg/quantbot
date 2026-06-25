"""
backtesting/crypto_backtest.py — Backtest the crypto screener "BUY when score >= 8" strategy.

Uses the same rule-based scorer as screener_backtest.py but with
crypto-specific thresholds (higher volatility, RSI bands adjusted).

STRATEGY:
  BUY  when: score >= 8
  SELL when: score drops below 5  OR  RSI > 75  OR  death cross
             OR  7-day loss > 15% (crypto stop-loss)

HOW TO RUN:
  python backtesting/crypto_backtest.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass
from loguru import logger
from backtesting.monte_carlo import run as monte_carlo_run, print_summary


# ─── Config ───────────────────────────────────────────────────────────────────

CRYPTO_WATCHLIST = [
    "BTC-USD", "ETH-USD", "SOL-USD",
    "DOGE-USD", "XRP-USD",
]
LOOKBACK_PERIOD   = "2y"
BUY_SCORE         = 8
SELL_SCORE        = 5
STOP_LOSS_PCT     = 0.15     # exit if position drops 15%
POSITION_SIZE_USD = 1000


# ─── Rule-based scorer (crypto-tuned) ─────────────────────────────────────────

def _score_row(row: pd.Series) -> int:
    score = 5

    # ── RSI (wider bands for crypto) ─────────────────────────────────────────
    rsi = row.get("rsi", 50)
    if rsi < 35:
        score += 2   # deeply oversold
    elif rsi < 50:
        score += 1   # healthy dip
    elif rsi > 75:
        score -= 2   # overbought
    elif rsi > 65:
        score -= 1

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd = row.get("macd_hist", 0)
    if macd > 0:
        score += 1
    elif macd < 0:
        score -= 1

    # ── Golden / death cross ─────────────────────────────────────────────────
    if row.get("golden_cross", False):
        score += 1
    elif row.get("death_cross", False):
        score -= 2

    # ── Relative volume ──────────────────────────────────────────────────────
    rel_vol = row.get("rel_volume", 1.0)
    if rel_vol > 2.0:
        score += 1
    elif rel_vol < 0.5:
        score -= 1

    # ── Bollinger Band position ───────────────────────────────────────────────
    bb_pos = row.get("bb_position", 0.5)
    if bb_pos < 0.15:
        score += 1   # near lower band — bounce candidate
    elif bb_pos > 0.95:
        score -= 1   # extremely stretched

    # ── 7-day momentum ───────────────────────────────────────────────────────
    chg_7d = row.get("price_change_7d", 0)
    if chg_7d > 0.10:
        score += 1   # strong recent momentum
    elif chg_7d < -0.15:
        score -= 1   # recent weakness

    return max(1, min(10, score))


# ─── Data + indicators ────────────────────────────────────────────────────────

def _load_and_compute(symbol: str) -> pd.DataFrame:
    df = _yf_dl(symbol, period=LOOKBACK_PERIOD, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    if len(df) < 60:
        return pd.DataFrame()

    close  = df["close"]
    volume = df["volume"]

    df["rsi"]       = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd_hist"] = ta.trend.MACD(close).macd_diff()
    df["ma50"]      = close.rolling(50).mean()
    df["ma200"]     = close.rolling(200).mean() if len(df) >= 200 else np.nan

    df["golden_cross"] = (
        (df["ma50"] > df["ma200"]) &
        (df["ma50"].shift(1) <= df["ma200"].shift(1))
    )
    df["death_cross"] = (
        (df["ma50"] < df["ma200"]) &
        (df["ma50"].shift(1) >= df["ma200"].shift(1))
    )

    df["avg_vol_20"] = volume.rolling(20).mean()
    df["rel_volume"] = volume / df["avg_vol_20"]

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_range = bb.bollinger_hband() - bb.bollinger_lband()
    df["bb_position"] = (close - bb.bollinger_lband()) / bb_range.replace(0, np.nan)

    df["price_change_7d"] = close.pct_change(7)

    df = df.dropna()
    df["score"] = df.apply(_score_row, axis=1)
    return df


# ─── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    pnl_pct:     float
    entry_score: int
    exit_reason: str


def _backtest_symbol(symbol: str) -> list[Trade]:
    df = _load_and_compute(symbol)
    if df.empty:
        logger.warning(f"{symbol}: not enough data, skipping.")
        return []

    label       = symbol.replace("-USD", "")
    trades      = []
    in_position = False
    entry_price = 0.0
    entry_date  = ""
    entry_score = 0

    for i in range(1, len(df)):
        row   = df.iloc[i]
        date  = str(df.index[i].date())
        score = int(row["score"])
        rsi   = float(row["rsi"])
        price = float(row["close"])

        if not in_position:
            if score >= BUY_SCORE:
                in_position = True
                entry_price = price
                entry_date  = date
                entry_score = score
                logger.debug(f"{label} BUY  {date}  ${entry_price:.4f}  score={score}")
        else:
            drawdown    = (price - entry_price) / entry_price
            exit_reason = None

            if drawdown <= -STOP_LOSS_PCT:
                exit_reason = f"Stop-loss ({drawdown:.1%})"
            elif score < SELL_SCORE:
                exit_reason = f"Score dropped to {score}"
            elif rsi > 75:
                exit_reason = "RSI overbought (>75)"
            elif row["death_cross"]:
                exit_reason = "Death cross"

            if exit_reason:
                pnl_pct = (price - entry_price) / entry_price
                trades.append(Trade(
                    symbol      = label,
                    entry_date  = entry_date,
                    exit_date   = date,
                    entry_price = entry_price,
                    exit_price  = price,
                    pnl_pct     = pnl_pct,
                    entry_score = entry_score,
                    exit_reason = exit_reason,
                ))
                logger.debug(
                    f"{label} SELL {date}  ${price:.4f}  "
                    f"P&L={pnl_pct:+.1%}  ({exit_reason})"
                )
                in_position = False

    return trades


# ─── Report ───────────────────────────────────────────────────────────────────

def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No trades generated — try lowering BUY_SCORE or extending the period.")
        return

    returns = [t.pnl_pct for t in all_trades]
    wins    = [t for t in all_trades if t.pnl_pct > 0]
    losses  = [t for t in all_trades if t.pnl_pct <= 0]

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Crypto Backtest  —  BUY when score >= {BUY_SCORE}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Coins tested    : {len(set(t.symbol for t in all_trades))}
 Total trades    : {len(all_trades)}
 Winners         : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers          : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win         : {np.mean([t.pnl_pct for t in wins]):+.1%}
 Avg loss        : {np.mean([t.pnl_pct for t in losses]):+.1%}
 Best trade      : {max(returns):+.1%}
 Worst trade     : {min(returns):+.1%}
 Avg return      : {np.mean(returns):+.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.symbol:<8}  {t.entry_date} → {t.exit_date}  "
            f"score={t.entry_score}  {t.pnl_pct:+6.1%}  ({t.exit_reason})"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(watchlist: list[str] = CRYPTO_WATCHLIST) -> list[float]:
    all_trades: list[Trade] = []
    for symbol in watchlist:
        logger.info(f"Backtesting {symbol}…")
        trades = _backtest_symbol(symbol)
        logger.info(f"{symbol}: {len(trades)} trades")
        all_trades.extend(trades)

    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    print("\nRunning Crypto Backtest  (score >= 8 = BUY)…")
    print(f"Coins  : {', '.join(s.replace('-USD','') for s in CRYPTO_WATCHLIST)}")
    print(f"Period : {LOOKBACK_PERIOD}\n")

    trade_returns = run_backtest(CRYPTO_WATCHLIST)

    if trade_returns:
        print("\n\nRunning 1,000-path Monte Carlo on backtest results…")
        result = monte_carlo_run(
            trade_returns,
            starting_equity=POSITION_SIZE_USD,
            n_runs=1000,
        )
        print_summary(result)
