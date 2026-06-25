"""
agents/crypto_mean_reversion.py — Live mean reversion screener for crypto.

BUY when:
  - Price at or below lower Bollinger Band (20-day, 2 std devs)
  - RSI(14) < 35  (deeply oversold)
  - Relative volume >= 1.5x  (capitulation / panic selling)

SELL signals are NOT handled here — exits are managed by the broker
via stop-loss orders placed at entry. The backtest showed 66% win rate
and +2.3% avg return with a -12% stop loss.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pandas as pd
import ta
from loguru import logger
from data.crypto_data import CRYPTO_WATCHLIST, get_crypto_ohlcv

RSI_OVERSOLD    = 40     # RSI must be below this (relaxed from 35 → catches early dips)
BB_PROXIMITY    = 0.02   # price within this % above lower band also qualifies
MIN_REL_VOLUME  = 1.2    # volume spike threshold (relaxed from 1.5 → less strict)
STOP_LOSS_PCT   = 0.12   # -12% stop loss (matches backtest)

# ── BTC regime gate ───────────────────────────────────────────────────────────
# Block ALL mean-reversion entries when BTC has fallen >10% over the last 7
# trading days.  When the whole market is in freefall, dip-buying becomes
# catching falling knives — the Feb-2026 cluster of stop-outs proved this.
BTC_DRAWDOWN_GATE  = -0.10   # gate threshold
BTC_LOOKBACK_DAYS  = 7


def _btc_7d_return() -> float:
    """
    Fetch BTC-USD and return the 7-trading-day percentage change in close price.
    Returns 0.0 on any fetch failure so the gate stays open (fail-open is safer
    than silently blocking all entries when yfinance is down).
    """
    try:
        import yfinance as yf
        df = yf.download("BTC-USD", period="30d", interval="1d",
                         progress=False, auto_adjust=True, timeout=10)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if len(df) < BTC_LOOKBACK_DAYS + 1:
            return 0.0
        close = df["close"].iloc
        ret   = float((close[-1] - close[-(BTC_LOOKBACK_DAYS + 1)]) / close[-(BTC_LOOKBACK_DAYS + 1)])
        return ret
    except Exception as exc:
        logger.warning(f"BTC regime check failed: {exc} — gate staying open")
        return 0.0


def _compute(symbol: str) -> dict:
    """Compute mean reversion indicators for a crypto symbol."""
    df = get_crypto_ohlcv(symbol, period="6mo")
    if df.empty or len(df) < 30:
        return {}

    close  = df["close"].squeeze()
    volume = df["volume"].squeeze()

    rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_lower  = float(bb.bollinger_lband().iloc[-1])
    bb_middle = float(bb.bollinger_mavg().iloc[-1])

    # Ultra-low-priced / high-vol coins (SHIB-USD: ~$0.000012) need two fixes:
    # 1. Non-positive band:   2*STD20 > MA20 after a sharp drop -> bb_lower <= 0
    #    -> price > 0 >= bb_lower means `at_lower_band` is always False AND the
    #    bb_pct_above calc in screen() divides by zero.
    # 2. Sub-cent precision:  rounding bb_lower to 4 dp underflows to 0.0 for
    #    SHIB-style prices, which would also cause the divzero downstream.
    # Reject the coin when bb_lower is non-positive OR would underflow at the
    # storage precision we use below (1e-8). Mean reversion on a coin with no
    # measurable lower-band distance is meaningless anyway.
    if bb_lower <= 1e-8 or bb_middle <= 1e-8:
        logger.debug(
            f"crypto_mean_reversion | {symbol}: bb_lower={bb_lower:.3e} "
            f"bb_middle={bb_middle:.3e} -- non-positive / sub-1e-8 band; skipping."
        )
        return {}

    avg_vol    = float(volume.rolling(20).mean().iloc[-1])
    rel_volume = round(float(volume.iloc[-1] / avg_vol), 2) if avg_vol else 1.0

    price = float(close.iloc[-1])
    # Storage precision: 8 decimals (not 4). Coins like SHIB-USD trade at
    # ~5e-6 USD and any 4-decimal round() underflows to 0.0, which then causes
    # divide-by-zero in screen()'s bb_pct_above calc downstream. 8 decimals is
    # well within Alpaca's reported price precision and preserves enough sig
    # figs for both BB math and stop/target display.
    return {
        "symbol":     symbol.replace("-USD", ""),
        "price":      round(price, 8),
        "rsi":        round(rsi, 2),
        "bb_lower":   round(bb_lower, 8),
        "bb_middle":  round(bb_middle, 8),
        "rel_volume": rel_volume,
        "stop":       round(price * (1 - STOP_LOSS_PCT), 8),
        "target":     round(bb_middle, 8),   # revert to 20MA
    }


def screen(watchlist: list[str] = CRYPTO_WATCHLIST) -> list[dict]:
    """
    Screen watchlist for mean reversion BUY setups.
    Returns only coins that meet all three entry conditions.

    BTC regime gate: if BTC has fallen >10% in the last 7 trading days,
    the entire crypto market is in freefall and no new entries are opened.
    """
    # ── BTC regime gate (checked once before scanning any coin) ──────────────
    btc_ret = _btc_7d_return()
    if btc_ret < BTC_DRAWDOWN_GATE:
        logger.warning(
            f"CryptoMeanRev GATE: BTC 7-day return {btc_ret:+.1%} < {BTC_DRAWDOWN_GATE:.0%} "
            f"— market in freefall, skipping all entries today"
        )
        return []
    logger.info(f"CryptoMeanRev: BTC 7-day return {btc_ret:+.1%} — gate open")

    signals = []
    for symbol in watchlist:
        logger.info(f"MeanRev screen | {symbol}...")
        try:
            d = _compute(symbol)
            if not d:
                continue

            at_lower_band = d["price"] <= d["bb_lower"] * (1 + BB_PROXIMITY)
            rsi_oversold  = d["rsi"] < RSI_OVERSOLD
            volume_spike  = d["rel_volume"] >= MIN_REL_VOLUME

            # How far each condition is from triggering (for status logging)
            bb_pct_above  = (d["price"] - d["bb_lower"]) / d["bb_lower"] * 100
            rsi_gap       = d["rsi"] - RSI_OVERSOLD
            vol_gap       = d["rel_volume"] - MIN_REL_VOLUME

            if at_lower_band and rsi_oversold and volume_spike:
                d["signal"] = "BUY"
                d["reason"] = (
                    f"Price ${d['price']} near/below lower BB ${d['bb_lower']}, "
                    f"RSI={d['rsi']} oversold, volume {d['rel_volume']}x avg"
                )
                signals.append(d)
                logger.info(
                    f"MeanRev BUY | {d['symbol']}  ${d['price']}  "
                    f"RSI={d['rsi']}  vol={d['rel_volume']}x  "
                    f"stop={d['stop']}  target={d['target']}"
                )
            else:
                # Always log at INFO so operator can see how close each coin is
                rsi_val = d["rsi"]
                vol_val = d["rel_volume"]
                bb_str  = "✓" if at_lower_band else f"✗ +{bb_pct_above:.1f}% above band"
                rsi_str = "✓" if rsi_oversold  else f"✗ {rsi_val:.0f} (need <{RSI_OVERSOLD})"
                vol_str = "✓" if volume_spike   else f"✗ {vol_val:.2f}x (need >{MIN_REL_VOLUME}x)"
                logger.info(
                    f"MeanRev WAIT | {d['symbol']:<6}  ${d['price']:>10.4f}  "
                    f"BB {bb_str}  RSI {rsi_str}  Vol {vol_str}"
                )
        except Exception as exc:
            logger.error(f"MeanRev screen {symbol} failed: {exc}")
        time.sleep(1)  # avoid yfinance rate limiting

    return signals


if __name__ == "__main__":
    results = screen()
    if results:
        print(f"\n{'Symbol':<8} {'Price':>10} {'RSI':>6} {'Vol':>6} {'Stop':>10} {'Target':>10}")
        print("-" * 58)
        for r in results:
            print(
                f"{r['symbol']:<8} ${r['price']:>9.4f}  {r['rsi']:>5.1f}  "
                f"{r['rel_volume']:>5.1f}x  ${r['stop']:>9.4f}  ${r['target']:>9.4f}"
            )
    else:
        print("No mean reversion setups found.")
