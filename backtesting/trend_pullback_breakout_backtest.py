"""
backtesting/trend_pullback_breakout_backtest.py — Trend Pullback Breakout strategy.

STRATEGY LOGIC (trend-continuation — orthogonal to VWAP/RSI mean-reversion):
  LONG when: 20 EMA > 50 EMA            (trend filter — uptrend established)
             AND price pulls back to/below the 20 EMA   (low <= ema20)
             AND current close > previous bar's high     (breakout confirmation)
             AND volume > 1.5x 20-period avg volume       (participation)

  SHORT: exact reverse of the above.

  EXIT:  2R target (risk:reward = 1:2), where risk = entry - stop
         Stop = the SAFER (wider) of:
            • below the pullback low (structural stop), OR
            • 1.5 ATR below entry
         This matches the spec: "Below the pullback low. Or 1.5 ATR below entry."

WHY THIS WORKS:
  The existing live strategies (Screener, Chart Pattern, Crypto MeanRev) are
  mean-reversion / pattern / score-based. This is pure trend-continuation, so
  its returns are largely uncorrelated — a genuine diversifier rather than a
  correlated bet. The EMA-cross trend filter keeps us on the right side of the
  market; the pullback-then-breakout entry buys strength after a dip rather
  than chasing extension.

RISK LIMITS (from config.py / .env):
  Position sizing respects MAX_POSITION_SIZE_USD. A daily-loss circuit breaker
  halts new entries once the simulated day is down DAILY_LOSS_LIMIT_USD.

HOW TO RUN:
  python backtesting/trend_pullback_breakout_backtest.py
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
EMA_FAST          = 20      # fast EMA (trend + pullback reference)
EMA_SLOW          = 50      # slow EMA (trend filter)
VOL_AVG_PERIOD    = 20      # average volume lookback
VOL_MULT          = 1.5     # volume must exceed avg × this
ATR_PERIOD        = 14
ATR_STOP_MULT     = 1.5     # ATR stop = entry -/+ (ATR × this)
RR_TARGET         = 2.0     # take-profit at risk × this (2R)
POSITION_SIZE_USD = 1000    # base notional per trade (Monte Carlo)

# Risk limits — sourced from config.py (falls back to .env defaults)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as _cfg  # noqa
    MAX_POSITION_SIZE_USD = float(getattr(_cfg, "MAX_POSITION_SIZE_USD", 500))
    DAILY_LOSS_LIMIT_USD  = float(getattr(_cfg, "DAILY_LOSS_LIMIT_USD", 100))
except Exception:
    MAX_POSITION_SIZE_USD = float(os.getenv("MAX_POSITION_SIZE_USD", 500))
    DAILY_LOSS_LIMIT_USD  = float(os.getenv("DAILY_LOSS_LIMIT_USD", 100))


# ─── Data + indicators ────────────────────────────────────────────────────────

def _load_and_compute(symbol: str) -> pd.DataFrame:
    df = _yf_dl(symbol, period=LOOKBACK_PERIOD, interval="1d", progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    if len(df) < EMA_SLOW + 10:
        return pd.DataFrame()

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    df["avgvol"]   = volume.rolling(VOL_AVG_PERIOD).mean()
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()

    return df.dropna()


# ─── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:       str
    direction:    str        # "long" or "short"
    entry_date:   str
    exit_date:    str
    entry_price:  float
    exit_price:   float
    stop_price:   float
    target_price: float
    pnl_pct:      float
    exit_reason:  str


def _backtest_symbol(symbol: str) -> list[Trade]:
    df = _load_and_compute(symbol)
    if df.empty:
        logger.warning(f"{symbol}: not enough data, skipping.")
        return []

    trades       = []
    in_position  = False
    direction    = 0          # +1 long, -1 short
    entry_price  = 0.0
    stop_price   = 0.0
    target_price = 0.0
    entry_date   = ""

    # daily-loss circuit breaker (in $ on POSITION_SIZE_USD notional)
    cur_day      = None
    day_pnl_usd  = 0.0

    for i in range(2, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]
        d     = df.index[i].date()
        date  = str(d)

        # reset daily P&L tracker
        if d != cur_day:
            cur_day, day_pnl_usd = d, 0.0

        close   = float(row["close"])
        op      = float(row["open"])
        emaf    = float(row["ema_fast"])
        emas    = float(row["ema_slow"])
        atr     = float(row["atr"])
        vol     = float(row["volume"])
        avgvol  = float(row["avgvol"])

        if not in_position:
            up    = emaf > emas
            down  = emaf < emas
            vol_ok = avgvol > 0 and vol > avgvol * VOL_MULT

            long_sig  = up   and (float(row["low"])  <= emaf) and (close > float(prev["high"])) and vol_ok
            short_sig = down and (float(row["high"]) >= emaf) and (close < float(prev["low"]))  and vol_ok

            # circuit breaker: no new entries if day already down past the limit
            if day_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
                long_sig = short_sig = False

            if long_sig:
                in_position  = True
                direction    = 1
                entry_price  = close
                entry_date   = date
                struct_stop  = min(float(prev["low"]), float(prev2["low"]))
                atr_stop     = close - atr * ATR_STOP_MULT
                stop_price   = min(struct_stop, atr_stop)          # safer (wider) stop
                risk         = entry_price - stop_price
                target_price = entry_price + risk * RR_TARGET
            elif short_sig:
                in_position  = True
                direction    = -1
                entry_price  = close
                entry_date   = date
                struct_stop  = max(float(prev["high"]), float(prev2["high"]))
                atr_stop     = close + atr * ATR_STOP_MULT
                stop_price   = max(struct_stop, atr_stop)
                risk         = stop_price - entry_price
                target_price = entry_price - risk * RR_TARGET

        else:
            exit_reason = None
            hi, lo = float(row["high"]), float(row["low"])

            if direction == 1:
                if lo <= stop_price:
                    exit_px, exit_reason = stop_price, f"Stop-loss (max of pullback-low / {ATR_STOP_MULT}ATR)"
                elif hi >= target_price:
                    exit_px, exit_reason = target_price, f"Take-profit ({RR_TARGET}R)"
            else:
                if hi >= stop_price:
                    exit_px, exit_reason = stop_price, f"Stop-loss (max of pullback-high / {ATR_STOP_MULT}ATR)"
                elif lo <= target_price:
                    exit_px, exit_reason = target_price, f"Take-profit ({RR_TARGET}R)"

            if exit_reason:
                pnl_pct = (exit_px - entry_price) / entry_price * direction
                day_pnl_usd += pnl_pct * POSITION_SIZE_USD
                trades.append(Trade(
                    symbol       = symbol,
                    direction    = "long" if direction == 1 else "short",
                    entry_date   = entry_date,
                    exit_date    = date,
                    entry_price  = entry_price,
                    exit_price   = exit_px,
                    stop_price   = stop_price,
                    target_price = target_price,
                    pnl_pct      = pnl_pct,
                    exit_reason  = exit_reason,
                ))
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
    longs   = [t for t in all_trades if t.direction == "long"]
    shorts  = [t for t in all_trades if t.direction == "short"]

    reasons: dict[str, int] = {}
    for t in all_trades:
        key = t.exit_reason.split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1

    avg_win  = np.mean([t.pnl_pct for t in wins])  if wins   else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    gross_win  = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Trend Pullback Breakout Backtest
 Entry: EMA{EMA_FAST}>EMA{EMA_SLOW} + pullback to EMA{EMA_FAST} + breakout + vol>{VOL_MULT}x
 Exit:  {RR_TARGET}R target  |  stop = safer of pullback-extreme / {ATR_STOP_MULT}ATR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Stocks tested   : {len(set(t.symbol for t in all_trades))}
 Total trades    : {len(all_trades)}   (long {len(longs)} / short {len(shorts)})
 Winners         : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers          : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win         : {avg_win:+.1%}
 Avg loss        : {avg_loss:+.1%}
 Risk/Reward     : {rr_ratio:.2f}:1
 Profit factor   : {pf:.2f}
 Best trade      : {max(returns):+.1%}
 Worst trade     : {min(returns):+.1%}
 Avg return      : {np.mean(returns):+.1%}
 Exit reasons    : {", ".join(f"{k} ({v})" for k, v in reasons.items())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(
            f"  {t.symbol:<6} {t.direction:<5} {t.entry_date} → {t.exit_date}  "
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
    print("\nRunning Trend Pullback Breakout Backtest…")
    print(f"Tickers : {', '.join(DEFAULT_TICKERS)}")
    print(f"Period  : {LOOKBACK_PERIOD}")
    print(f"Entry   : EMA{EMA_FAST}>EMA{EMA_SLOW} + pullback + breakout + vol>{VOL_MULT}x")
    print(f"Exit    : {RR_TARGET}R target | stop=safer(pullback-extreme, {ATR_STOP_MULT}ATR)")
    print(f"Risk    : max pos ${MAX_POSITION_SIZE_USD:.0f} | daily loss limit ${DAILY_LOSS_LIMIT_USD:.0f}\n")

    trade_returns = run_backtest(DEFAULT_TICKERS)

    if trade_returns:
        print("\n\nRunning 1,000-path Monte Carlo on backtest results...")
        result = monte_carlo_run(
            trade_returns,
            starting_equity=POSITION_SIZE_USD,
            n_runs=1000,
        )
        print_summary(result)
