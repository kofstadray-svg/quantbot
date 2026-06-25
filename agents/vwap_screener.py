"""
agents/vwap_screener.py — VWAP Pullback + RSI(2) live screener.

Rule-based (no Claude API call needed) — signals fire instantly.

BUY when:
  - Price > 20-day rolling VWAP  (institutional sentiment positive)
  - Price > 50-day MA            (uptrend intact)
  - RSI(2) < 30                  (short-term oversold pullback)

Auto-trades BUY signals via Alpaca ($100 per trade).
Stop/target levels are logged for reference.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ta
from loguru import logger
from data.market_data import get_ohlcv

from data.custom_watchlist import CUSTOM_WATCHLIST
DEFAULT_WATCHLIST = CUSTOM_WATCHLIST

RSI2_BUY        = 30    # RSI(2) must be below this
MA_TREND_PERIOD = 50    # price must be above this MA
ATR_STOP_MULT   = 1.0   # stop = entry - (ATR × this)
ATR_TARGET_MULT = 1.5   # target = entry + (ATR × this)


def _compute(ticker: str) -> dict:
    """Compute VWAP, RSI(2), ATR, and 50MA for a ticker."""
    df = get_ohlcv(ticker, period="6mo")
    if df.empty or len(df) < MA_TREND_PERIOD + 10:
        return {}

    close  = df["close"].squeeze()
    high   = df["high"].squeeze()
    low    = df["low"].squeeze()
    volume = df["volume"].squeeze()

    # RSI(2)
    rsi2 = float(ta.momentum.RSIIndicator(close, window=2).rsi().iloc[-1])

    # 50-day MA
    ma50 = float(close.rolling(MA_TREND_PERIOD).mean().iloc[-1])

    # Rolling 20-day VWAP approximation
    typical_price = (high + low + close) / 3
    vwap = float(
        (typical_price * volume).rolling(20).sum().iloc[-1]
        / volume.rolling(20).sum().iloc[-1]
    )

    # ATR (14-day)
    atr = float(
        ta.volatility.AverageTrueRange(high, low, close, window=14)
        .average_true_range().iloc[-1]
    )

    price = float(close.iloc[-1])
    return {
        "ticker":  ticker,
        "price":   round(price, 2),
        "rsi2":    round(rsi2, 1),
        "ma50":    round(ma50, 2),
        "vwap":    round(vwap, 2),
        "atr":     round(atr, 4),
        "stop":    round(price - atr * ATR_STOP_MULT, 2),
        "target":  round(price + atr * ATR_TARGET_MULT, 2),
    }


def screen(watchlist: list[str] = DEFAULT_WATCHLIST) -> list[dict]:
    """
    Screen watchlist for VWAP Pullback + RSI(2) BUY signals.
    Returns list of dicts — only BUY signals included (empty = no setups today).
    """
    signals = []
    for ticker in watchlist:
        logger.info(f"VWAP screen | {ticker}...")
        try:
            d = _compute(ticker)
            if not d:
                continue

            above_vwap = d["price"] > d["vwap"]
            above_ma50 = d["price"] > d["ma50"]
            rsi_dip    = d["rsi2"] < RSI2_BUY

            if above_vwap and above_ma50 and rsi_dip:
                d["signal"] = "BUY"
                d["reason"] = (
                    f"Price ${d['price']} above VWAP ${d['vwap']} & 50MA ${d['ma50']}, "
                    f"RSI(2)={d['rsi2']} oversold"
                )
                signals.append(d)
                logger.info(
                    f"VWAP BUY | {ticker}  ${d['price']}  "
                    f"RSI2={d['rsi2']}  stop={d['stop']}  target={d['target']}"
                )
            else:
                logger.debug(
                    f"VWAP SKIP | {ticker}  price={d['price']}  vwap={d['vwap']}  "
                    f"ma50={d['ma50']}  rsi2={d['rsi2']}  "
                    f"[above_vwap={above_vwap} above_ma50={above_ma50} rsi_dip={rsi_dip}]"
                )
        except Exception as exc:
            logger.error(f"VWAP screen {ticker} failed: {exc}")

    return signals


if __name__ == "__main__":
    results = screen()
    if results:
        print(f"\n{'Ticker':<8} {'Price':>8} {'RSI2':>6} {'VWAP':>8} {'Stop':>8} {'Target':>8}")
        print("-" * 56)
        for r in results:
            print(
                f"{r['ticker']:<8} ${r['price']:>7.2f}  {r['rsi2']:>5.1f}  "
                f"${r['vwap']:>7.2f}  ${r['stop']:>7.2f}  ${r['target']:>7.2f}"
            )
    else:
        print("No VWAP Pullback setups found today.")
