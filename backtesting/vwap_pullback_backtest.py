"""
backtesting/vwap_pullback_backtest.py — VWAP Pullback + RSI(2) strategy backtest.

STRATEGY LOGIC:
  BUY  when: price is ABOVE the daily VWAP (overall trend is up)
             AND RSI(2) < 30 (short-term pullback / oversold dip)
             AND price > 50-day MA (longer-term uptrend confirmation)

  SELL when: price reaches 1.5x ATR above entry (take profit)
             OR  1.0x ATR below entry (stop loss)
             OR  RSI(2) > 80 (momentum exhausted)
             OR  price closes below VWAP (trend broken)

WHY THIS WORKS:
  VWAP is the institutional benchmark — when price is above VWAP the big
  money is net-long that day. RSI(2) is an ultra-short mean-reversion
  indicator; dipping below 30 while above VWAP identifies high-probability
  "buy the dip" moments. ATR-based exits adapt to each stock's volatility.

APPROXIMATE VWAP (daily bars):
  True intraday VWAP requires tick data. On daily bars we approximate it
  as (High + Low + Close) / 3, which is the "typical price" and a close
  proxy for the volume-weighted average on a daily timeframe.

HOW TO RUN:
  python backtesting/vwap_pullback_backtest.py
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

DEFAULT_TICKERS = [
    "AAPL", "NVDA", "MSFT", "META", "TSLA", "AMZN", "GOOGL",
    "SPY",  "QQQ",  "LIFE", "RDVT", "CLFD",
]
LOOKBACK_PERIOD   = "2y"
RSI2_BUY          = 30     # RSI(2) below this = oversold pullback — enter
RSI2_SELL         = 80     # RSI(2) above this = momentum exhausted — exit
ATR_STOP_MULT     = 1.0    # stop loss = entry - (ATR * this)
ATR_TARGET_MULT   = 1.5    # take profit = entry + (ATR * this)
MA_TREND_PERIOD   = 50     # price must be above this MA to buy
POSITION_SIZE_USD = 1000


# ─── Data + indicators ────────────────────────────────────────────────────────

def _load_and_compute(symbol: str) -> pd.DataFrame:
    df = _yf_dl(symbol, period=LOOKBACK_PERIOD, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    if len(df) < MA_TREND_PERIOD + 10:
        return pd.DataFrame()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # RSI(2) — ultra-short mean reversion oscillator
    df["rsi2"] = ta.momentum.RSIIndicator(close, window=2).rsi()

    # 50-day MA for trend filter
    df["ma50"] = close.rolling(MA_TREND_PERIOD).mean()

    # Daily VWAP approximation: (H+L+C)/3 rolling cumulative
    # Using expanding typical-price × volume / expanding volume (resets daily)
    # On daily bars: rolling 20-day VWAP is a practical proxy
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    df["vwap"] = tp_vol.rolling(20).sum() / volume.rolling(20).sum()

    # ATR (14-day) for dynamic stop/target
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    return df.dropna()


# ─── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    stop_price:  float
    target_price: float
    pnl_pct:     float
    exit_reason: str


def _backtest_symbol(symbol: str) -> list[Trade]:
    df = _load_and_compute(symbol)
    if df.empty:
        logger.warning(f"{symbol}: not enough data, skipping.")
        return []

    trades      = []
    in_position = False
    entry_price = 0.0
    stop_price  = 0.0
    target_price = 0.0
    entry_date  = ""

    for i in range(1, len(df)):
        row   = df.iloc[i]
        date  = str(df.index[i].date())
        price = float(row["close"])
        rsi2  = float(row["rsi2"])
        vwap  = float(row["vwap"])
        ma50  = float(row["ma50"])
        atr   = float(row["atr"])

        if not in_position:
            # BUY conditions:
            #   1. Price above VWAP     (institutional sentiment positive)
            #   2. Price above 50MA     (longer-term uptrend intact)
            #   3. RSI(2) < 30          (short-term oversold pullback)
            above_vwap = price > vwap
            above_ma50 = price > ma50
            rsi_dip    = rsi2 < RSI2_BUY

            if above_vwap and above_ma50 and rsi_dip:
                in_position  = True
                entry_price  = price
                entry_date   = date
                stop_price   = price - (atr * ATR_STOP_MULT)
                target_price = price + (atr * ATR_TARGET_MULT)
                logger.debug(
                    f"{symbol} BUY  {date}  ${price:.2f}  "
                    f"RSI2={rsi2:.1f}  VWAP={vwap:.2f}  "
                    f"stop={stop_price:.2f}  target={target_price:.2f}"
                )
        else:
            exit_reason = None

            # EXIT: ATR-based stop loss
            if price <= stop_price:
                exit_reason = f"Stop-loss (ATR×{ATR_STOP_MULT})"
            # EXIT: ATR-based take profit
            elif price >= target_price:
                exit_reason = f"Take-profit (ATR×{ATR_TARGET_MULT})"
            # EXIT: RSI(2) overbought — momentum exhausted
            elif rsi2 > RSI2_SELL:
                exit_reason = f"RSI(2) exhausted ({rsi2:.0f})"
            # EXIT: price drops below VWAP — trend broken
            elif price < vwap:
                exit_reason = "Price below VWAP"

            if exit_reason:
                pnl_pct = (price - entry_price) / entry_price
                trades.append(Trade(
                    symbol       = symbol,
                    entry_date   = entry_date,
                    exit_date    = date,
                    entry_price  = entry_price,
                    exit_price   = price,
                    stop_price   = stop_price,
                    target_price = target_price,
                    pnl_pct      = pnl_pct,
                    exit_reason  = exit_reason,
                ))
                logger.debug(
                    f"{symbol} SELL {date}  ${price:.2f}  "
                    f"P&L={pnl_pct:+.1%}  ({exit_reason})"
                )
                in_position = False

    return trades


# ─── Report ───────────────────────────────────────────────────────────────────

def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No trades generated — conditions may not have been met in test period.")
        return

    returns = [t.pnl_pct for t in all_trades]
    wins    = [t for t in all_trades if t.pnl_pct > 0]
    losses  = [t for t in all_trades if t.pnl_pct <= 0]

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in all_trades:
        key = t.exit_reason.split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1

    # Risk/reward
    avg_win  = np.mean([t.pnl_pct for t in wins])  if wins   else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 VWAP Pullback + RSI(2) Backtest
 Entry: Price > VWAP + Price > 50MA + RSI(2) < {RSI2_BUY}
 Exit:  ATR stop×{ATR_STOP_MULT}  |  ATR target×{ATR_TARGET_MULT}  |  RSI(2)>{RSI2_SELL}  |  <VWAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Stocks tested   : {len(set(t.symbol for t in all_trades))}
 Total trades    : {len(all_trades)}
 Winners         : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers          : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win         : {avg_win:+.1%}
 Avg loss        : {avg_loss:+.1%}
 Risk/Reward     : {rr_ratio:.2f}:1
 Best trade      : {max(returns):+.1%}
 Worst trade     : {min(returns):+.1%}
 Avg return      : {np.mean(returns):+.1%}
 Exit reasons    : {", ".join(f"{k} ({v})" for k, v in reasons.items())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.symbol:<6}  {t.entry_date} → {t.exit_date}  "
            f"{t.pnl_pct:+6.1%}  ({t.exit_reason})"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(tickers: list[str] = DEFAULT_TICKERS) -> list[float]:
    all_trades: list[Trade] = []
    for symbol in tickers:
        logger.info(f"Backtesting {symbol}…")
        trades = _backtest_symbol(symbol)
        logger.info(f"{symbol}: {len(trades)} trades")
        all_trades.extend(trades)

    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    print("\nRunning VWAP Pullback + RSI(2) Backtest…")
    print(f"Tickers : {', '.join(DEFAULT_TICKERS)}")
    print(f"Period  : {LOOKBACK_PERIOD}")
    print(f"Entry   : Price > VWAP + Price > {MA_TREND_PERIOD}MA + RSI(2) < {RSI2_BUY}")
    print(f"Exit    : ATR×{ATR_STOP_MULT} stop  |  ATR×{ATR_TARGET_MULT} target  |  RSI(2)>{RSI2_SELL}  |  Below VWAP\n")

    trade_returns = run_backtest(DEFAULT_TICKERS)

    if trade_returns:
        print("\n\nRunning 1,000-path Monte Carlo on backtest results...")
        result = monte_carlo_run(
            trade_returns,
            starting_equity=POSITION_SIZE_USD,
            n_runs=1000,
        )
        print_summary(result)
