"""
backtesting/crypto_mean_reversion.py — Mean Reversion crypto strategy backtest.

STRATEGY LOGIC (opposite of trend-following):
  BUY  when: price drops 2+ std devs below 20-day MA (Bollinger lower band)
             AND RSI < 35 (oversold)
             AND volume spike > 1.5x average (capitulation/panic selling)

  SELL when: price reverts to 20-day MA (middle Bollinger Band)
             OR RSI crosses above 60 (momentum restored)
             OR stop-loss hit (-12%)

WHY THIS WORKS FOR CRYPTO:
  Crypto overcorrects hard during fear/panic. When a coin drops 2 standard
  deviations below its mean WITH high volume, it's often a capitulation wick
  that snaps back quickly. This strategy harvests those bounces.

HOW TO RUN:
  python backtesting/crypto_mean_reversion.py
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
LOOKBACK_PERIOD      = "2y"
RSI_OVERSOLD         = 40      # buy when RSI below this (relaxed from 35)
BB_PROXIMITY         = 0.02    # price within 2% above lower band also qualifies
RSI_RECOVERED        = 60      # sell when RSI recovers to this
MIN_REL_VOLUME       = 1.2     # volume must be 1.2x average (relaxed from 1.5x)
STOP_LOSS_PCT        = 0.12    # hard stop at -12%
POSITION_SIZE_USD    = 1000

# ── BTC regime gate ───────────────────────────────────────────────────────────
# When BTC falls >10% in the last 7 calendar days, the whole crypto market is
# in freefall — mean reversion entries become catching falling knives.
# The Feb-2026 cluster of stop-outs (BTC, ETH, SOL, XRP, AVAX all -12–20%)
# occurred in a single 4-day window when BTC dropped ~18%.  This gate would
# have blocked every one of those entries.
BTC_DRAWDOWN_GATE  = -0.10   # block entries when BTC 7-day return < -10%
BTC_LOOKBACK_DAYS  = 7


# ─── BTC regime calendar ─────────────────────────────────────────────────────

def _build_btc_regime(period: str = LOOKBACK_PERIOD) -> dict[str, bool]:
    """
    Return a {date_str: entry_ok} dict for every trading day in the period.

    entry_ok = True  → BTC 7-day return >= BTC_DRAWDOWN_GATE  (safe to enter)
    entry_ok = False → BTC crashed >10% in last 7 days  (gate is closed)

    This is computed from BTC-USD closes so it applies even when the coin
    being backtested is not BTC itself.
    """
    df = _yf_dl("BTC-USD", period=period, interval="1d",
                progress=False, auto_adjust=True, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    close = df["close"]
    # 7-day return: today vs. 7 trading days ago
    ret_7d = close.pct_change(BTC_LOOKBACK_DAYS)

    regime: dict[str, bool] = {}
    for dt, ret in ret_7d.items():
        date_str = str(pd.Timestamp(dt).date())
        regime[date_str] = bool(float(ret) >= BTC_DRAWDOWN_GATE) if not pd.isna(ret) else True

    return regime


# ─── Data + indicators ────────────────────────────────────────────────────────

def _load_and_compute(symbol: str) -> pd.DataFrame:
    df = _yf_dl(symbol, period=LOOKBACK_PERIOD, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    if len(df) < 30:
        return pd.DataFrame()

    close  = df["close"]
    volume = df["volume"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Bollinger Bands (20-day, 2 std devs)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()   # 20-day MA = mean reversion target
    df["bb_upper"]  = bb.bollinger_hband()

    # Relative volume
    df["avg_vol_20"] = volume.rolling(20).mean()
    df["rel_volume"] = volume / df["avg_vol_20"]

    return df.dropna()


# ─── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    pnl_pct:     float
    exit_reason: str


def _backtest_symbol(
    symbol: str,
    btc_regime: dict[str, bool] | None = None,
) -> list[Trade]:
    """
    Backtest mean reversion on a single symbol.

    btc_regime : optional {date_str: entry_ok} dict built by _build_btc_regime().
                 When provided, entries are blocked on days where BTC has fallen
                 more than BTC_DRAWDOWN_GATE over the prior 7 days.
    """
    df = _load_and_compute(symbol)
    if df.empty:
        logger.warning(f"{symbol}: not enough data, skipping.")
        return []

    label       = symbol.replace("-USD", "")
    trades      = []
    in_position = False
    entry_price = 0.0
    entry_date  = ""
    gated_count = 0   # how many entries were blocked by the BTC regime gate

    for i in range(1, len(df)):
        row   = df.iloc[i]
        date  = str(df.index[i].date())
        price = float(row["close"])
        rsi   = float(row["rsi"])

        if not in_position:
            # BUY: price at/near lower Bollinger Band + oversold RSI + volume spike
            at_lower_band  = price <= float(row["bb_lower"]) * (1 + BB_PROXIMITY)
            rsi_oversold   = rsi < RSI_OVERSOLD
            volume_spike   = float(row["rel_volume"]) >= MIN_REL_VOLUME

            # ── BTC regime gate ───────────────────────────────────────────────
            # If BTC has crashed >10% in the last 7 days, skip the entry.
            # The whole market is in freefall; mean reversion doesn't work.
            if at_lower_band and rsi_oversold and volume_spike and btc_regime is not None:
                if not btc_regime.get(date, True):
                    gated_count += 1
                    logger.debug(
                        f"{label} GATE  {date}  BTC 7d return < {BTC_DRAWDOWN_GATE:.0%} — skipped"
                    )
                    continue  # skip this entry

            if at_lower_band and rsi_oversold and volume_spike:
                in_position = True
                entry_price = price
                entry_date  = date
                logger.debug(
                    f"{label} BUY  {date}  ${price:.4f}  "
                    f"RSI={rsi:.1f}  vol={row['rel_volume']:.1f}x"
                )
        else:
            drawdown    = (price - entry_price) / entry_price
            exit_reason = None

            # SELL: price reverts to 20-day MA
            if price >= float(row["bb_middle"]):
                exit_reason = "Reverted to mean (20MA)"
            # SELL: RSI momentum restored
            elif rsi >= RSI_RECOVERED:
                exit_reason = f"RSI recovered to {rsi:.0f}"
            # SELL: stop-loss
            elif drawdown <= -STOP_LOSS_PCT:
                exit_reason = f"Stop-loss ({drawdown:.1%})"

            if exit_reason:
                pnl_pct = (price - entry_price) / entry_price
                trades.append(Trade(
                    symbol      = label,
                    entry_date  = entry_date,
                    exit_date   = date,
                    entry_price = entry_price,
                    exit_price  = price,
                    pnl_pct     = pnl_pct,
                    exit_reason = exit_reason,
                ))
                logger.debug(
                    f"{label} SELL {date}  ${price:.4f}  "
                    f"P&L={pnl_pct:+.1%}  ({exit_reason})"
                )
                in_position = False

    if gated_count:
        logger.info(f"{label}: {gated_count} entries blocked by BTC regime gate")

    return trades


# ─── Report ───────────────────────────────────────────────────────────────────

def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No trades generated — market conditions may not have triggered entries.")
        return

    returns = [t.pnl_pct for t in all_trades]
    wins    = [t for t in all_trades if t.pnl_pct > 0]
    losses  = [t for t in all_trades if t.pnl_pct <= 0]

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in all_trades:
        key = t.exit_reason.split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Crypto Mean Reversion Backtest
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
 Exit reasons    : {", ".join(f"{k} ({v})" for k, v in reasons.items())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.symbol:<8}  {t.entry_date} → {t.exit_date}  "
            f"{t.pnl_pct:+6.1%}  ({t.exit_reason})"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(watchlist: list[str] = CRYPTO_WATCHLIST) -> list[float]:
    # Build BTC regime calendar once — reused for every coin in the watchlist.
    logger.info("Building BTC regime calendar…")
    btc_regime = _build_btc_regime()
    closed_days = sum(1 for ok in btc_regime.values() if not ok)
    logger.info(
        f"BTC regime: {closed_days} gated days out of {len(btc_regime)} "
        f"({closed_days/max(len(btc_regime),1):.1%} of trading days blocked)"
    )

    all_trades: list[Trade] = []
    for symbol in watchlist:
        logger.info(f"Backtesting {symbol}…")
        trades = _backtest_symbol(symbol, btc_regime=btc_regime)
        logger.info(f"{symbol}: {len(trades)} trades")
        all_trades.extend(trades)

    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    print("\nRunning Crypto Mean Reversion Backtest…")
    print(f"Coins  : {', '.join(s.replace('-USD','') for s in CRYPTO_WATCHLIST)}")
    print(f"Period : {LOOKBACK_PERIOD}")
    print(f"Entry  : Price at lower Bollinger Band + RSI < {RSI_OVERSOLD} + volume > {MIN_REL_VOLUME}x")
    print(f"Exit   : Revert to 20MA  OR  RSI > {RSI_RECOVERED}  OR  stop-loss -{STOP_LOSS_PCT:.0%}\n")

    trade_returns = run_backtest(CRYPTO_WATCHLIST)

    if trade_returns:
        print("\n\nRunning 1,000-path Monte Carlo on backtest results…")
        result = monte_carlo_run(
            trade_returns,
            starting_equity=POSITION_SIZE_USD,
            n_runs=1000,
        )
        print_summary(result)
