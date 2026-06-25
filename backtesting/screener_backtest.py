"""
backtesting/screener_backtest.py — Backtest the AI stock screener strategy.

Uses a rule-based scorer that EXACTLY mirrors the current stock_screener.py
orthogonal feature set (post-redundancy-elimination):

  TREND     : golden cross (50MA > 200MA) · MA50 slope · ADX(14)
  MOMENTUM  : RSI(14) only
  VOLATILITY: ATR percentile rank (0-100) — compression is a setup
  LIQUIDITY : relative volume · OBV slope · RS vs SPY

STRATEGY:
  BUY  when: score >= 8
  SELL when: score < 5
        OR  RSI > 75
        OR  stop loss hit (1 ATR below entry)
        OR  time stop (3 days, no momentum: RSI2 < 50 and close < EMA5)

HOW TO RUN:
  python backtesting/screener_backtest.py
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

LOOKBACK_PERIOD   = "2y"
BUY_SCORE         = 8      # matches live screener threshold
SELL_SCORE        = 5
TIME_STOP_DAYS    = 3
POSITION_SIZE_USD = 100

# ── indicator helpers ─────────────────────────────────────────────────────────

def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX(14) series — trend strength, direction-agnostic."""
    adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period, fillna=False)
    return adx_ind.adx()


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low,
         (high - close.shift()).abs(),
         (low  - close.shift()).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _atr_pct_rank_series(high: pd.Series, low: pd.Series,
                          close: pd.Series, period: int = 14) -> pd.Series:
    """Rolling percentile rank of ATR within its own 252-bar history (0-100)."""
    atr = _atr_series(high, low, close, period)
    ranks = atr.rolling(252, min_periods=60).apply(
        lambda w: float((w[:-1] < w[-1]).sum()) / max(1, len(w) - 1) * 100,
        raw=True
    )
    return ranks


def _obv_slope_series(close: pd.Series, volume: pd.Series, lookback: int = 10) -> pd.Series:
    """Linear-regression slope of OBV over `lookback` bars (+ / -)."""
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv[i] = obv[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv[i] = obv[i - 1] - volume.iloc[i]
        else:
            obv[i] = obv[i - 1]
    obv_s = pd.Series(obv, index=close.index)
    return obv_s.rolling(lookback).apply(
        lambda w: float(np.polyfit(range(len(w)), w, 1)[0]), raw=True
    )


def _ma50_slope_series(close: pd.Series, bars: int = 10) -> pd.Series:
    """Slope of 50MA over `bars` bars, expressed as % of price per bar."""
    ma50 = close.rolling(50).mean()
    slope = ma50.rolling(bars).apply(
        lambda w: float(np.polyfit(range(len(w)), w, 1)[0]), raw=True
    )
    return slope / close * 100


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    return ta.momentum.RSIIndicator(close=close, window=period).rsi()


def _rsi2_series(close: pd.Series) -> pd.Series:
    return ta.momentum.RSIIndicator(close=close, window=2).rsi()


# ── scoring ───────────────────────────────────────────────────────────────────

def _score_row(row: pd.Series) -> float:
    """
    Orthogonal factor scoring — mirrors live stock_screener.py SYSTEM_PROMPT.

    Dimensions (each ±1.0 to ±2.0):
      trend_score     : golden_cross + ma50_slope + ADX
      momentum_score  : RSI only
      volatility_score: ATR percentile (compression = setup)
      liquidity_score : rel_vol + obv_slope + rs_vs_spy
    """
    score = 5.0   # neutral baseline

    # ── TREND ─────────────────────────────────────────────────────────────────
    golden  = bool(row.get("golden_cross", False))
    slope   = float(row.get("ma50_slope_pct", 0.0))
    adx     = float(row.get("adx_14", 20.0))

    if golden:   score += 0.8
    else:        score -= 0.8
    if slope > 0.05:  score += 0.5
    elif slope < -0.05: score -= 0.5
    if adx > 25:  score += 0.5
    elif adx < 15: score -= 0.2

    # ── MOMENTUM (RSI only) ───────────────────────────────────────────────────
    rsi = float(row.get("rsi_14", 50.0))
    if rsi < 30:          score += 2.0   # oversold bounce
    elif rsi < 40:        score += 1.0
    elif 40 <= rsi < 60:  score += 0.0   # neutral
    elif 60 <= rsi < 75:  score -= 0.5   # extended but not extreme
    else:                  score -= 1.5   # overbought

    # ── VOLATILITY — ATR percentile ───────────────────────────────────────────
    atr_rank = float(row.get("atr_pct_rank", 50.0))
    if atr_rank < 20:    score += 0.8    # compressed → coiled spring
    elif atr_rank > 75:  score -= 0.8   # expanded → chasing volatility

    # ── LIQUIDITY ─────────────────────────────────────────────────────────────
    rel_vol  = float(row.get("rel_volume", 1.0))
    obv_s    = float(row.get("obv_slope", 0.0))

    if rel_vol > 1.5:    score += 0.8
    elif rel_vol < 0.7:  score -= 0.5
    if obv_s > 0:        score += 0.5
    elif obv_s < 0:      score -= 0.5

    return max(1.0, min(10.0, score))


# ── data loading + feature engineering ───────────────────────────────────────

def _load_and_compute(ticker: str) -> pd.DataFrame:
    raw = _yf_dl(ticker, period=LOOKBACK_PERIOD, interval="1d",
                 auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 220:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["close"]  = raw["Close"].squeeze()
    df["high"]   = raw["High"].squeeze()
    df["low"]    = raw["Low"].squeeze()
    df["volume"] = raw["Volume"].squeeze()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # Trend
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["golden_cross"]  = (ma50 > ma200).astype(float)
    df["ma50_slope_pct"] = _ma50_slope_series(close)
    df["adx_14"]        = _adx_series(high, low, close)

    # Momentum
    df["rsi_14"] = _rsi_series(close)
    df["rsi_2"]  = _rsi2_series(close)

    # Volatility
    df["atr_14"]       = _atr_series(high, low, close)
    df["atr_pct_rank"] = _atr_pct_rank_series(high, low, close)

    # Liquidity
    vol_avg_20      = volume.rolling(20).mean()
    df["rel_volume"] = (volume / vol_avg_20.replace(0, np.nan)).fillna(1.0)
    df["obv_slope"]  = _obv_slope_series(close, volume)

    df = df.dropna()
    df["score"] = df.apply(_score_row, axis=1)
    return df


# ── trade simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float; pnl_pct: float
    hold_days: int; entry_score: float; exit_reason: str


def _backtest_ticker(ticker: str) -> list[Trade]:
    df = _load_and_compute(ticker)
    if df.empty:
        return []

    trades: list[Trade] = []
    in_pos      = False
    entry_price = entry_idx = 0
    entry_date  = ""
    entry_score = 0.0
    entry_atr   = 0.0

    ema5 = df["close"].ewm(span=5, adjust=False).mean()

    for i in range(1, len(df)):
        row   = df.iloc[i]
        date  = str(df.index[i].date())
        score = float(row["score"])
        rsi   = float(row["rsi_14"])
        rsi2  = float(row.get("rsi_2", 50.0))
        price = float(row["close"])
        atr   = float(row["atr_14"])
        ema5v = float(ema5.iloc[i])

        if not in_pos:
            if score >= BUY_SCORE:
                in_pos      = True
                entry_price = price
                entry_date  = date
                entry_score = score
                entry_idx   = i
                entry_atr   = atr
        else:
            days_held = i - entry_idx
            dd        = (price - entry_price) / entry_price
            hard_stop = entry_price - entry_atr   # 1 ATR stop

            reason = None

            # Priority 1 — hard stop (1 ATR)
            if price < hard_stop:
                reason = f"Hard stop -ATR ({dd:.1%})"

            # Priority 2 — time stop (3 days, no momentum)
            elif days_held >= TIME_STOP_DAYS and rsi2 < 50 and price < ema5v:
                reason = f"Time stop day {days_held} ({dd:.1%})"

            # Priority 3 — RSI overbought
            elif rsi > 75:
                reason = f"RSI overbought ({rsi:.0f})"

            # Priority 4 — score collapsed
            elif score < SELL_SCORE:
                reason = f"Score dropped to {score:.1f}"

            if reason:
                trades.append(Trade(
                    ticker, entry_date, date, entry_price, price,
                    (price - entry_price) / entry_price,
                    days_held, entry_score, reason
                ))
                in_pos = False

    return trades


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_report(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("No trades generated.")
        return

    returns  = [t.pnl_pct for t in all_trades]
    wins     = [t for t in all_trades if t.pnl_pct > 0]
    losses   = [t for t in all_trades if t.pnl_pct <= 0]
    avg_hold = np.mean([t.hold_days for t in all_trades])
    avg_win  = np.mean([t.pnl_pct for t in wins])  if wins   else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss else float("inf")
    exp      = (len(wins) / len(all_trades)) * avg_win + (len(losses) / len(all_trades)) * avg_loss
    reasons: dict[str, int] = {}
    for t in all_trades:
        k = t.exit_reason.split("(")[0].strip()
        reasons[k] = reasons.get(k, 0) + 1

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AI Stock Screener Backtest  (score >= {BUY_SCORE})
 Indicators: RSI · ADX · ATR%ile · OBV · RelVol (orthogonal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Tickers tested  : {len(set(t.ticker for t in all_trades))}
 Total trades    : {len(all_trades)}
 Winners         : {len(wins)}  ({len(wins)/len(all_trades):.0%})
 Losers          : {len(losses)}  ({len(losses)/len(all_trades):.0%})
 Avg win         : {avg_win:+.1%}
 Avg loss        : {avg_loss:+.1%}
 Risk/Reward     : {rr:.2f}:1
 Expectancy/trade: {exp:+.2%}
 Avg hold        : {avg_hold:.0f} days
 Best trade      : {max(returns):+.1%}
 Worst trade     : {min(returns):+.1%}
 Avg return      : {np.mean(returns):+.1%}
 Exit reasons    : {", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Trade log (best → worst):
""")
    for t in sorted(all_trades, key=lambda x: x.pnl_pct, reverse=True):
        print(f"  {t.ticker:6s}  {t.entry_date} → {t.exit_date}  "
              f"score={t.entry_score:.1f}  hold={t.hold_days:3d}d  "
              f"{t.pnl_pct:+6.1%}  ({t.exit_reason})")


def run_backtest(tickers: list[str]) -> list[float]:
    all_trades: list[Trade] = []
    for ticker in tickers:
        logger.info(f"Screener: {ticker}…")
        try:
            trades = _backtest_ticker(ticker)
        except Exception as exc:
            logger.warning(f"  {ticker}: skipped — {exc}")
            trades = []
        if trades:
            logger.info(f"  {ticker}: {len(trades)} trades")
        all_trades.extend(trades)
    _print_report(all_trades)
    return [t.pnl_pct for t in all_trades]


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist
    tickers = list(dict.fromkeys(CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=20)))
    print(f"\nRunning Orthogonal Screener Backtest on {len(tickers)} tickers…")
    print(f"Period: {LOOKBACK_PERIOD}  |  BUY score >= {BUY_SCORE}\n")
    returns = run_backtest(tickers)
    if returns:
        print("\nRunning Monte Carlo…")
        result = monte_carlo_run(returns, starting_equity=POSITION_SIZE_USD, n_runs=1000)
        print_summary(result)
