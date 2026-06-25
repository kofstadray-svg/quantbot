"""
agents/exit_manager.py -- Regime-aware active exit management.

Exit behaviour adapts to the current market regime of each individual stock,
rather than applying a fixed set of correlated momentum indicators.

Regime classification (per ticker, daily data):
  Regime           Classification
  ---------------  ------------------------------------------------
  trending         ADX(14) >= 25 AND 20-EMA slope positive
  choppy           ADX(14) <  25 OR  20-EMA slope flat/negative
  high_vol         ATR(14) / close >= 2.0%
  normal_vol       ATR(14) / close  1.0-2.0%
  low_vol          ATR(14) / close <  1.0%

Regime -> exit parameters:
  Regime                  ATR mult  Trail init  Time days
  ----------------------  --------  ----------  ---------
  trending + low_vol        1.5x      3.0%        5
  trending + normal         1.75x     4.0%        5
  trending + high_vol       2.0x      5.0%        6
  choppy + low_vol          0.75x     1.5%        2
  choppy + normal           1.0x      2.0%        3
  choppy + high_vol         1.25x     3.0%        4

Profit ladder (2 rungs):
  Rung 1: +3%  -> sell 25%  (trail widens to 1x, was 2x initially)
  Rung 2: +8%  -> sell 25%  (trail tightens to 0.6x)
  Rest  : trail remaining 50% with adaptive width

Trailing stop is ALWAYS active from bar 1 (not only after first partial).
  Before partial1: trail = 2.0x regime trail_pct  (room to breathe)
  After  partial1: trail = 1.0x regime trail_pct  (normal)
  After  partial2: trail = 0.6x regime trail_pct  (lock in gains)

Breakout failure detection (first BREAKOUT_FAIL_DAYS trading days only):
  If gain_pct < -1x ATR_pct within the first 3 days, the gap/breakout thesis
  has reversed before the full hard stop would fire.  Exit immediately.

Chandelier Exit (volatility-adaptive trail from the peak):
  Standalone stop in the cascade: close < highest_high_since_entry - ATR*mult
  (mult = 3.0 trending / 2.0 choppy). Trails from the true peak (highest HIGH)
  with an ATR-scaled distance, so winners run but excessive give-back exits.
  Runner portion (after both partials): trails on a tighter 2.5x Chandelier.

Trend signals (one per regime type, not all-at-once):
  TRENDING  -> Supertrend(10,3) flip, then 20-EMA cross as confirmation
  CHOPPY    -> RSI mean-reversion: RSI peak >= 55, now falls below 40
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
from utils.metrics import counter as _counter
from utils.indicators import supertrend_state
from utils.retry import with_retry

# Throttle "API blind" alerts so an extended outage doesn't spam Telegram.
_LAST_BLIND_ALERT = {"ts": 0.0}
_BLIND_ALERT_COOLDOWN_SEC = 1800   # at most one alert per 30 min per source

def _alert_blind(source: str, exc: Exception) -> None:
    """Send a throttled Telegram alert that a component is API-blind (fail-closed)."""
    import time as _t
    now = _t.time()
    if now - _LAST_BLIND_ALERT["ts"] < _BLIND_ALERT_COOLDOWN_SEC:
        return
    _LAST_BLIND_ALERT["ts"] = now
    msg = (
        f"⚠️ <b>{source} is API-blind</b>\n"
        f"Could not reach Alpaca after retries: {type(exc).__name__}\n"
        f"Failing closed — positions are UNMANAGED until the API recovers. "
        f"No action taken on stale data."
    )
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            import requests as _req
            _req.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
                timeout=5,
            )
    except Exception:
        # Telegram itself may be down during an outage; the log error above is
        # the guaranteed signal. Never let alerting failure break the caller.
        logger.error(f"{source} API-blind alert could not be sent (Telegram unreachable?).")

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Profit ladder rungs
PARTIAL1_PCT  = 0.03   # +3%  -> sell 25%
PARTIAL2_PCT  = 0.08   # +8%  -> sell another 25%

# ── Institutional Breakout ATR exit profile ──────────────────────────────────
# ATR-based profit ladder for institutional_breakout positions.
# Entry stop: 2 ATR below entry (set at order time, stored in exit_state.json)
# TP1: price rises 2 ATR above entry → sell 50% of position
# TP2: price rises 4 ATR above entry → sell 25% of position
# Runner (25%): trail with Chandelier Exit (2.5 ATR from peak)
IB_PARTIAL1_ATR = 2.0   # sell 50% at +2 ATR profit
IB_PARTIAL2_ATR = 4.0   # sell 25% at +4 ATR profit
IB_TRAIL_MULT   = 2.5   # Chandelier trail mult for runner

# Trail width multipliers by phase
TRAIL_MULT_PRE_P1  = 2.0   # before first partial: wide trail (let it run)
TRAIL_MULT_POST_P1 = 1.0   # after first partial:  normal width
TRAIL_MULT_POST_P2 = 0.6   # after second partial: tight (lock in gains)

# Breakout failure window (trading days from entry)
BREAKOUT_FAIL_DAYS     = 3
BREAKOUT_FAIL_ATR_MULT = 1.0   # fires if close < entry - 1x ATR (faster than hard stop)

# Chandelier Exit: trails from highest-high-since-entry minus ATR * multiplier.
CHANDELIER_LOOKBACK    = 22    # bars for the highest-high window (when no entry date)
CHANDELIER_MULT_TREND  = 3.0   # classic value for trending regimes (lets winners run)
CHANDELIER_MULT_CHOPPY = 2.0   # tighter in choppy regimes (give back less)
CHANDELIER_MULT_RUNNER = 2.5   # runner portion after both partials (lock the trend)

# Event-aware tightening (news + SEC filings via OpenBB).
# When a held name has just had material news or filed an 8-K, the stop is
# tighter because the stock is much more likely to gap-and-reverse around the
# event. The factor scales both the hard-stop ATR multiplier and the trail
# width — so a 0.7 factor on a 1.75x ATR stop becomes 1.225x, AND a 4% trail
# becomes 2.8%. Both close the risk window at the same time.
NEWS_WITHIN_HOURS  = 6     # only "news" if published in the last 6 hours
TIGHTEN_NEWS       = 0.7   # 30% closer stop & trail on recent news
TIGHTEN_8K         = 0.6   # 40% closer stop & trail on 8-K filing day


def _has_recent_news(ticker: str, within_hours: int = NEWS_WITHIN_HOURS) -> bool:
    """True iff FinnHub reports a news item for this ticker within `within_hours`.
    Falls back to False on any error — never raises.
    OpenBB dependency removed; now uses data.finnhub_data.
    """
    try:
        from config import FINNHUB_ENABLED
        if not FINNHUB_ENABLED:
            return False
        from data.finnhub_data import get_company_news
        articles = get_company_news(ticker, days=1)
        if not articles:
            return False
        cutoff = datetime.now(timezone.utc).timestamp() - within_hours * 3600
        for a in articles:
            ts = a.get("datetime", 0)
            if ts and float(ts) > cutoff:
                return True
    except Exception as exc:
        logger.debug(f"ExitMgr | _has_recent_news({ticker}) failed: {exc}")
    return False


def _filed_8k_today(ticker: str) -> bool:
    """Check for 8-K filing today via FMP (OpenBB dependency removed).
    Returns False if FMP is not configured or any error occurs.
    """
    try:
        from config import FMP_ENABLED
        if not FMP_ENABLED:
            return False
        from data.fmp_data import get_earnings_date
        # FMP doesn't expose 8-K directly on starter tier — stub for now.
        # A proper implementation would call /sec_filings endpoint.
        return False
    except Exception:
        return False


def _event_tighten_factor(ticker: str, is_crypto: bool) -> tuple[float, str]:
    """Return (factor, reason). factor in [0.6, 1.0]; 1.0 means no change.

    Crypto is skipped entirely (no SEC / no useful yfinance news coverage).
    An 8-K filing takes precedence over generic news (it is the harder signal).
    Any failure inside OpenBB returns (1.0, "") so this never tightens unless
    we are confident an event happened.
    """
    if is_crypto:
        return 1.0, ""
    try:
        if _filed_8k_today(ticker):
            return TIGHTEN_8K, "8-K filed today"
        if _has_recent_news(ticker):
            return TIGHTEN_NEWS, f"news within {NEWS_WITHIN_HOURS}h"
    except Exception as exc:
        logger.debug(f"ExitMgr | event-tighten check failed for {ticker}: {exc}")
    return 1.0, ""


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------
_STATE_PATH = Path(__file__).parent.parent / "data" / "exit_state.json"


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _init_ticker_state(state: dict, ticker: str, entry_price: float) -> None:
    if ticker not in state:
        state[ticker] = {
            "entry_date":    date.today().isoformat(),
            "entry_price":   entry_price,
            "partial_taken":  False,   # rung 1 taken (+3%)
            "partial2_taken": False,   # rung 2 taken (+8%)
            "trail_high":    entry_price,
            "rsi_peak":      0.0,      # tracks highest RSI seen for mean-rev signal
        }
    else:
        # Back-fill new fields into existing state entries
        ts = state[ticker]
        ts.setdefault("partial2_taken", False)
        ts.setdefault("rsi_peak",       0.0)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    vals = tr.rolling(period).mean().dropna()
    return float(vals.iloc[-1]) if len(vals) else float((high - low).mean())


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi_s = 100 - (100 / (1 + rs))
    vals  = rsi_s.dropna()
    return float(vals.iloc[-1]) if len(vals) else 50.0


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index (Wilder smoothing)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm >= plus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14     = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    minus_di  = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    denom     = (plus_di + minus_di).replace(0, float("nan"))
    dx        = 100 * (plus_di - minus_di).abs() / denom
    adx_s     = dx.ewm(alpha=1 / period, adjust=False).mean().dropna()
    return float(adx_s.iloc[-1]) if len(adx_s) else 20.0


def _trading_days_held(entry_date_str: str) -> int:
    try:
        entry = date.fromisoformat(entry_date_str)
    except (ValueError, TypeError):
        return 0
    today = date.today()
    days, cur = 0, entry + timedelta(days=1)
    while cur <= today:
        if cur.weekday() < 5:
            days += 1
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitRegime:
    name:       str
    trend:      Literal["trending", "choppy"]
    vol:        Literal["low_vol", "normal_vol", "high_vol"]
    atr_mult:   float   # hard-stop = entry - atr_mult * ATR
    trail_pct:  float   # base trailing stop width (scaled by phase multiplier)
    time_days:  int     # days held before time stop fires
    adx:        float
    atr_pct:    float


def classify_exit_regime(df: pd.DataFrame, market=None) -> ExitRegime:
    """
    Classify the exit regime for a single ticker using its daily OHLCV.

    The per-stock regime (this stock's own ADX/ATR) is the primary driver, but
    it is now contextualised by the market-wide regime so the two systems agree
    on direction rather than appearing to contradict each other:

      - The "high_vol" ATR cutoff floats with the market vol regime. A 2% ATR
        stock is genuinely elevated in a calm tape but ordinary when the whole
        market is volatile, so the cutoff rises in high/extreme-vol markets.
      - In a confirmed market uptrend (strong_bull/bull, risk_on) we give
        trending names slightly more room (wider time stop) so healthy winners
        in a strong tape are not cut prematurely.
      - In a risk_off market we tighten: trending stops behave one notch closer
        to choppy, protecting gains when market internals weaken.

    `market` is an optional market_regime.RegimeState. When None, behaviour is
    identical to the original per-stock-only classification.
    """
    close  = df["close"]
    atr    = _atr(df)
    adx    = _adx(df)
    last   = float(close.iloc[-1])
    atr_pct = atr / last if last > 0 else 0.015

    ema20 = _ema(close, 20)
    ema_slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-6])) / float(ema20.iloc[-6])         if len(ema20) >= 6 and float(ema20.iloc[-6]) > 0 else 0.0
    is_trending = adx >= 25 and ema_slope > 0.001

    trend: Literal["trending", "choppy"] = "trending" if is_trending else "choppy"

    # Market-adaptive high-vol cutoff: baseline 2%, raised when the broad market
    # is itself in a high/extreme vol regime (so per-stock labels don't all flip
    # to high_vol just because the whole tape is wide that week).
    high_vol_cut = 0.02
    norm_vol_cut = 0.01
    market_vol = getattr(market, "vol_regime", None)
    if market_vol == "high":
        high_vol_cut, norm_vol_cut = 0.028, 0.014
    elif market_vol == "extreme":
        high_vol_cut, norm_vol_cut = 0.035, 0.018

    if atr_pct >= high_vol_cut:
        vol: Literal["low_vol", "normal_vol", "high_vol"] = "high_vol"
    elif atr_pct >= norm_vol_cut:
        vol = "normal_vol"
    else:
        vol = "low_vol"

    PARAMS = {
        ("trending", "low_vol"):    (1.50, 0.030, 5),
        ("trending", "normal_vol"): (1.75, 0.040, 5),
        ("trending", "high_vol"):   (2.00, 0.050, 6),
        ("choppy",   "low_vol"):    (0.75, 0.015, 2),
        ("choppy",   "normal_vol"): (1.00, 0.020, 3),
        ("choppy",   "high_vol"):   (1.25, 0.030, 4),
    }
    atr_mult, trail_pct, time_days = PARAMS[(trend, vol)]

    # Market-context overlay on the per-stock parameters.
    market_trend = getattr(market, "trend_regime", None)
    market_risk  = getattr(market, "risk_regime", None)
    if trend == "trending" and market_trend in ("strong_bull", "bull") and market_risk == "risk_on":
        # Healthy stock in a confirmed uptrend: give winners one extra day.
        time_days += 1
    elif market_risk == "risk_off":
        # Market internals weak: protect gains, tighten one notch.
        atr_mult  = round(atr_mult * 0.85, 3)
        time_days = max(2, time_days - 1)

    return ExitRegime(
        name=f"{trend}_{vol}",
        trend=trend, vol=vol,
        atr_mult=atr_mult, trail_pct=trail_pct, time_days=time_days,
        adx=round(adx, 1), atr_pct=round(atr_pct, 4),
    )


# ---------------------------------------------------------------------------
# Exit evaluators
# ---------------------------------------------------------------------------

def _check_hard_stop(
    close: float, entry_price: float, atr: float, atr_mult: float
) -> tuple[bool, str]:
    stop = entry_price - atr_mult * atr
    if close < stop:
        return True, (
            f"hard stop: close={close:.2f} < entry-{atr_mult}xATR={stop:.2f} "
            f"(entry={entry_price:.2f}  ATR={atr:.2f})"
        )
    return False, ""


def _check_breakout_failure(
    close: float, entry_price: float, atr: float, days_held: int
) -> tuple[bool, str]:
    """
    Early-exit check for breakout / gap trades in the first BREAKOUT_FAIL_DAYS.

    Fires when: days_held <= BREAKOUT_FAIL_DAYS AND
                close < entry_price - BREAKOUT_FAIL_ATR_MULT * ATR

    This is tighter than the regime hard stop (which uses 0.75x-2.0x ATR)
    and fires faster -- a 1xATR reversal in 3 days means the breakout is failing,
    not just normal noise.  Only active in the setup window; after that the
    normal hard stop takes over.
    """
    if days_held > BREAKOUT_FAIL_DAYS:
        return False, ""
    fail_level = entry_price - BREAKOUT_FAIL_ATR_MULT * atr
    if close <= fail_level:
        return True, (
            f"breakout failure (day {days_held}/{BREAKOUT_FAIL_DAYS}): "
            f"close={close:.2f} < entry-{BREAKOUT_FAIL_ATR_MULT}xATR={fail_level:.2f} "
            f"-- gap/breakout reversing, exiting before full hard stop"
        )
    return False, ""


def _check_time_stop(
    close: float, df: pd.DataFrame, days_held: int, time_days: int
) -> tuple[bool, str]:
    """
    Regime-aware time stop: cut after time_days if no momentum.
    Uses 5-EMA as the momentum proxy.
    """
    if days_held < time_days:
        return False, ""
    ema5 = float(_ema(df["close"], 5).iloc[-1])
    if close < ema5:
        return True, (
            f"time stop: {days_held} days held (threshold={time_days}), "
            f"close={close:.2f} < EMA5={ema5:.2f} -- no momentum"
        )
    return False, ""


def _check_trending_exit(close: float, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Trending regime exit: 20-EMA cross.
    Close drops below the 20-day EMA -> trend is over.
    """
    ema20 = float(_ema(df["close"], 20).iloc[-1])
    if close < ema20:
        return True, f"20-EMA cross: close={close:.2f} < EMA20={ema20:.2f}"
    return False, ""


def _check_supertrend_exit(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Supertrend(10,3) exit: the trailing ATR band flipped to bearish.

    Supertrend is a cleaner trend-follower than the raw 20-EMA cross -- its band
    only flips after a decisive ATR-scaled move, so it cuts whipsaw exits while
    still catching genuine trend reversals. When price closes below the line
    (direction = -1), the uptrend that justified holding has ended.
    """
    st = supertrend_state(df["high"], df["low"], df["close"], period=10, multiplier=3.0)
    if st["direction"] == -1 and st["value"] is not None:
        flipped = " (just flipped)" if st["just_flipped"] else ""
        return True, (
            f"Supertrend(10,3) bearish{flipped}: close below ST={st['value']:.2f}"
        )
    return False, ""


def _chandelier_stop(df: pd.DataFrame, entry_date_str: str, atr: float,
                     mult: float, lookback: int = CHANDELIER_LOOKBACK) -> float | None:
    """
    Chandelier Exit level = highest_high(since entry, capped at `lookback` bars)
                            - ATR * mult.

    Trails from the highest HIGH (the true peak, not the highest close), and the
    stop distance is ATR-scaled so it adapts to each stock's volatility -- wide
    rope for volatile names, tighter for calm ones. Returns the stop price, or
    None if uncomputable.
    """
    try:
        highs = df["high"]
        # Anchor the highest-high window to the entry date when we have it, so
        # the Chandelier trails from the peak reached DURING the trade, not from
        # an older pre-entry high. Fall back to a fixed lookback otherwise.
        hh = None
        if entry_date_str:
            try:
                idx_dates = pd.to_datetime(df.index).date
                entry_d = date.fromisoformat(entry_date_str)
                mask = [d >= entry_d for d in idx_dates]
                since_entry = highs[mask]
                if len(since_entry) >= 1:
                    hh = float(since_entry.max())
            except Exception:
                hh = None
        if hh is None:
            hh = float(highs.iloc[-lookback:].max())
        if math.isnan(hh) or atr <= 0:
            return None
        return hh - atr * mult
    except Exception:
        return None


def _check_chandelier_exit(close: float, df: pd.DataFrame, entry_date_str: str,
                           atr: float, mult: float) -> tuple[bool, str]:
    """
    Chandelier Exit: close below (highest_high_since_entry - ATR * mult).
    A volatility-adaptive trailing stop anchored to the peak. Classic mult = 3.0
    in trending regimes; tightened in choppy regimes (passed in by the caller).
    """
    stop = _chandelier_stop(df, entry_date_str, atr, mult)
    if stop is not None and close < stop:
        return True, (
            f"Chandelier exit: close={close:.2f} < HH-{mult}xATR={stop:.2f} "
            f"(ATR={atr:.2f}) -- gave back too much from the peak"
        )
    return False, ""


def _check_choppy_exit(
    close: float, df: pd.DataFrame, state_ticker: dict
) -> tuple[bool, str]:
    """
    Choppy regime exit: RSI mean-reversion exhaustion.
    RSI(14) previously reached >= 55, now falls below 40 -> bounce exhausted.
    """
    rsi_now  = _rsi(df["close"])
    rsi_peak = state_ticker.get("rsi_peak", 0.0)

    if rsi_now > rsi_peak:
        state_ticker["rsi_peak"] = rsi_now
        rsi_peak = rsi_now

    if rsi_peak >= 55 and rsi_now < 40:
        return True, (
            f"RSI mean-reversion: peak={rsi_peak:.0f} -> now={rsi_now:.0f} "
            f"(bounce exhausted)"
        )
    return False, ""


def _check_ib_profit_ladder(
    ticker: str,
    close: float,
    qty: float,
    state: dict,
    atr: float,
    df: "pd.DataFrame",
) -> tuple[str, float, str]:
    """
    ATR-based profit ladder for institutional_breakout positions.

    Phase       Trigger         Sell    Exit type
    ---------   -----------     ----    ---------------------
    Pre-TP1     --              --      2-ATR initial stop (checked in hard_stop)
    TP1         +2 ATR profit   50%     Partial sell
    TP2         +4 ATR profit   25%     Partial sell
    Runner      post-TP2        25%     Chandelier(2.5 ATR) trail

    Returns (action, sell_qty, reason).
    """
    ts             = state.get(ticker, {})
    entry_price    = ts.get("entry_price", close)
    partial_taken  = ts.get("ib_partial1_taken",  False)
    partial2_taken = ts.get("ib_partial2_taken", False)
    trail_high     = ts.get("trail_high", entry_price)
    if close > trail_high:
        state[ticker]["trail_high"] = close
        trail_high = close

    profit = close - entry_price

    # TP1: +2 ATR -> sell 50%
    if not partial_taken and profit >= IB_PARTIAL1_ATR * atr:
        sell_qty = round(qty * 0.50, 6)
        state[ticker]["ib_partial1_taken"] = True
        if sell_qty < 0.001:
            return "hold", 0.0, ""
        return "partial", sell_qty, (
            f"IB TP1: +{profit:.2f} >= +{IB_PARTIAL1_ATR}xATR={IB_PARTIAL1_ATR*atr:.2f}  "
            f"selling 50% ({sell_qty:.6g} shares)"
        )

    # TP2: +4 ATR -> sell 25%
    if partial_taken and not partial2_taken and profit >= IB_PARTIAL2_ATR * atr:
        sell_qty = round(qty * 0.25, 6)
        state[ticker]["ib_partial2_taken"] = True
        if sell_qty < 0.001:
            return "hold", 0.0, ""
        return "partial2", sell_qty, (
            f"IB TP2: +{profit:.2f} >= +{IB_PARTIAL2_ATR}xATR={IB_PARTIAL2_ATR*atr:.2f}  "
            f"selling 25% ({sell_qty:.6g} shares)"
        )

    # Runner: Chandelier trail after both partials
    if partial2_taken:
        entry_date_str = ts.get("entry_date", "")
        stop = _chandelier_stop(df, entry_date_str, atr, IB_TRAIL_MULT)
        if stop is not None and close < stop:
            sell_qty = min(qty, qty if not hasattr(qty, "__len__") else qty)
            return "trail", sell_qty, (
                f"IB runner Chandelier({IB_TRAIL_MULT}x): "
                f"close={close:.2f} < HH-{IB_TRAIL_MULT}xATR={stop:.2f}"
            )

    return "hold", 0.0, ""


def _check_profit_ladder(
    ticker: str,
    close: float,
    qty: float,
    state: dict,
    trail_pct: float,
) -> tuple[str, float, str]:
    """
    Two-rung profit ladder with always-on, phase-adaptive trailing stop.

    Phase         Trigger     Sell    Trail width
    ----------    -------     ----    --------------------------
    Pre-rung1     --          --      2.0x trail_pct  (wide, let it run)
    Post-rung1    >= +3%      25%     1.0x trail_pct  (normal)
    Post-rung2    >= +8%      25%     0.6x trail_pct  (tight, lock in gains)

    Returns (action, sell_qty, reason): action in {'partial','partial2','trail','hold'}.

    The trailing stop is ALWAYS active -- not only after a partial is taken.
    trail_high tracks the highest close seen since entry.
    """
    ts             = state.get(ticker, {})
    entry_price    = ts.get("entry_price", close)
    partial_taken  = ts.get("partial_taken",  False)
    partial2_taken = ts.get("partial2_taken", False)
    trail_high     = ts.get("trail_high",     entry_price)
    gain_pct       = (close / entry_price) - 1.0

    # Always update the trail high
    if close > trail_high:
        state[ticker]["trail_high"] = close
        trail_high = close

    # Determine current phase multiplier
    if partial2_taken:
        trail_mult = TRAIL_MULT_POST_P2   # 0.6x -- tight
    elif partial_taken:
        trail_mult = TRAIL_MULT_POST_P1   # 1.0x -- normal
    else:
        trail_mult = TRAIL_MULT_PRE_P1    # 2.0x -- wide (give it room)

    effective_trail = trail_pct * trail_mult

    # ---- Rung 1: +3% -> sell 25% ----
    if not partial_taken and gain_pct >= PARTIAL1_PCT:
        # Use fractional-safe quantity: round to 6dp, no floor/max(1) which
        # breaks on sub-4-share (fractional) positions.
        sell_qty = round(qty * 0.25, 6)
        state[ticker]["partial_taken"] = True
        if sell_qty < 0.001:
            # Position too small to split -- mark rung taken, skip the order
            logger.debug(
                f"ExitMgr | {ticker} rung-1 triggered but sell_qty={sell_qty:.6f} "
                f"< 0.001 minimum -- marking taken, holding full position"
            )
            return "hold", 0.0, ""
        return "partial", sell_qty, (
            f"profit ladder rung 1: +{gain_pct:.1%} >= +{PARTIAL1_PCT:.0%}, "
            f"taking 25% ({sell_qty:.6g} shares) -- trail narrows to 1x"
        )

    # ---- Rung 2: +8% -> sell another 25% ----
    if partial_taken and not partial2_taken and gain_pct >= PARTIAL2_PCT:
        sell_qty = round(qty * 0.25, 6)
        state[ticker]["partial2_taken"] = True
        if sell_qty < 0.001:
            logger.debug(
                f"ExitMgr | {ticker} rung-2 triggered but sell_qty={sell_qty:.6f} "
                f"< 0.001 minimum -- marking taken, holding full position"
            )
            return "hold", 0.0, ""
        return "partial2", sell_qty, (
            f"profit ladder rung 2: +{gain_pct:.1%} >= +{PARTIAL2_PCT:.0%}, "
            f"taking another 25% ({sell_qty:.6g} shares) -- trail tightens to 0.6x"
        )

    # ---- Always-on trailing stop ----
    # Floor: never trail below breakeven (prevents stop below entry when pre-partial)
    trail_floor = entry_price * (1 - effective_trail)
    trail_stop  = max(trail_floor, trail_high * (1 - effective_trail))

    if close < trail_stop:
        phase = "pre-rung1" if not partial_taken else ("post-rung2" if partial2_taken else "post-rung1")
        return "trail", qty, (
            f"trail stop [{phase}]: close={close:.2f} < trail={trail_stop:.2f} "
            f"(peak={trail_high:.2f}  width={effective_trail:.1%}={trail_mult}x{trail_pct:.1%})"
        )

    return "hold", 0.0, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exit_manager() -> int:
    """
    Evaluate regime-aware exit conditions for every open equity position.
    Returns number of exit orders placed.
    """
    from brokers.alpaca import market_sell, crypto_sell
    from data.market_data import get_ohlcv
    from data.crypto_data import get_crypto_ohlcv

    # Fetch the market-wide regime once (cached 30 min) so per-stock exit
    # classification is contextualised by broad-market conditions instead of
    # being computed in a vacuum. Failure is non-fatal — falls back to None,
    # which preserves the original per-stock-only behaviour.
    market_regime = None
    try:
        from agents.market_regime import compute_regime
        market_regime = compute_regime()
    except Exception as _mre:
        logger.debug(f"ExitMgr | market regime unavailable ({_mre}); per-stock only.")

    # Equities trade only during RTH; crypto trades 24/7. We compute whether
    # the equity session is open and use it to gate ONLY equity positions —
    # crypto positions are always evaluated below.
    equities_open = True
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5:
            equities_open = False
        else:
            open_t  = now.replace(hour=9,  minute=35, second=0, microsecond=0)
            close_t = now.replace(hour=15, minute=55, second=0, microsecond=0)
            if not (open_t <= now <= close_t):
                equities_open = False
    except ImportError:
        pass

    from alpaca.trading.client import TradingClient
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)

    # FAIL-CLOSED on position data. If we cannot reach Alpaca to read live
    # positions, we must NOT fall back to cached/stale position data -- acting
    # on stale holdings causes phantom sells / double-sells. Instead we skip the
    # cycle entirely and ALERT loudly, so the operator knows the exit manager is
    # blind (open positions are unmanaged until the API recovers). This is the
    # opposite policy from market-data caching, which is safe to serve stale.
    try:
        positions = with_retry(max_attempts=3, base_delay=2.0)(client.get_all_positions)()
    except Exception as exc:
        logger.error(
            f"ExitMgr | FAIL-CLOSED: cannot fetch positions after retries ({exc}). "
            f"Exit manager is BLIND this cycle -- open positions are UNMANAGED "
            f"until the API recovers. Not acting on stale data."
        )
        _alert_blind("Exit manager", exc)
        return 0

    if not positions:
        logger.debug("ExitMgr | no open positions.")
        return 0

    state = _load_state()
    exits_placed = 0

    for pos in positions:
        ticker = pos.symbol
        qty    = float(pos.qty)
        entry  = float(pos.avg_entry_price)
        # Use avg_entry_price as fallback only; real close comes from OHLCV below

        # Shares actually free to sell. Alpaca locks shares behind any open
        # (unfilled) sell order as "held_for_orders", so qty_available < qty when
        # a prior exit is still working. Selling the full qty in that case is
        # rejected with code 40310000 ("insufficient qty available"). If nothing
        # is free, an exit order is already pending — skip this position this
        # cycle and let the working order fill.
        try:
            qty_available = abs(float(getattr(pos, "qty_available", qty)))
        except (TypeError, ValueError):
            qty_available = abs(qty)
        if qty_available <= 0:
            logger.info(
                f"ExitMgr | {ticker}: 0 shares available ({qty} held for open "
                f"orders) — exit already pending, skipping this cycle."
            )
            continue

        # Detect crypto: Alpaca reports crypto positions as "BTC/USD" or "BTCUSD"
        # depending on API surface. Normalize to a yfinance symbol ("BTC-USD")
        # for OHLCV, and remember to route the sell through crypto_sell().
        is_crypto = ("/" in ticker) or (
            ticker.endswith("USD") and not ticker.endswith("-USD") and len(ticker) <= 8
        )

        # Gate equities to regular trading hours; crypto runs 24/7.
        if not is_crypto and not equities_open:
            logger.debug(f"ExitMgr | {ticker}: equities closed -- skipping.")
            continue

        _init_ticker_state(state, ticker, entry)

        try:
            if is_crypto:
                # Normalize to BASE/QUOTE, then build the yfinance symbol
                # ("AVAX/USD" -> "AVAX-USD"). Handles BTCUSD/ETHUSDT correctly.
                from brokers.alpaca import _normalize_crypto_symbol
                norm = _normalize_crypto_symbol(ticker)        # e.g. "AVAX/USD"
                yf_symbol = norm.replace("/", "-")             # e.g. "AVAX-USD"
                df = get_crypto_ohlcv(yf_symbol, period="3mo", interval="1d")
            else:
                df = get_ohlcv(ticker, period="3mo", interval="1d")
            if df is None or len(df) < 20:
                logger.warning(f"ExitMgr | {ticker}: insufficient data, skipping.")
                continue
        except Exception as exc:
            logger.warning(f"ExitMgr | {ticker}: data fetch failed: {exc}")
            continue

        # Use the OHLCV close as the authoritative price — more reliable than
        # pos.current_price which can be hours stale when fetched outside market hours.
        close = float(df["close"].iloc[-1])

        regime    = classify_exit_regime(df, market=market_regime)
        atr       = _atr(df)
        days_held = _trading_days_held(state[ticker].get("entry_date", ""))
        gain_pct  = (close / entry) - 1.0

        # Event-aware tightening: when a held name has fresh news (≤6h) or filed
        # an 8-K today, both the hard-stop ATR multiplier and the trail width get
        # multiplied by a sub-1 factor. The cost of being wrong is small (you
        # exit a position that recovers); the cost of being right (avoiding a
        # post-news gap-down on something you held) is large. Crypto is skipped.
        event_factor, event_reason = _event_tighten_factor(ticker, is_crypto)
        effective_atr_mult  = round(regime.atr_mult  * event_factor, 4)
        effective_trail_pct = round(regime.trail_pct * event_factor, 6)
        if event_factor < 1.0:
            logger.info(
                f"ExitMgr | {ticker} EVENT TIGHTENING x{event_factor:.2f} "
                f"({event_reason})  "
                f"atr_mult {regime.atr_mult:.2f} -> {effective_atr_mult:.2f}, "
                f"trail {regime.trail_pct:.1%} -> {effective_trail_pct:.1%}"
            )

        logger.debug(
            f"ExitMgr | {ticker} regime={regime.name}  "
            f"ADX={regime.adx}  ATR%={regime.atr_pct:.2%}  "
            f"atr_mult={effective_atr_mult}x  trail_base={effective_trail_pct:.1%}  "
            f"time_stop={regime.time_days}d  days_held={days_held}  "
            f"gain={gain_pct:+.2%}  "
            f"mkt={getattr(market_regime, 'label', 'n/a')}"
            f"{'  EVT=' + event_reason if event_factor < 1.0 else ''}"
        )

        exit_all    = False
        exit_reason = ""

        # 1. Breakout failure (early window: first BREAKOUT_FAIL_DAYS days)
        fired, reason = _check_breakout_failure(close, entry, atr, days_held)
        if fired:
            exit_all, exit_reason = True, reason

        # 2. Hard Stop (regime-scaled ATR multiplier, event-tightened)
        if not exit_all:
            fired, reason = _check_hard_stop(close, entry, atr, effective_atr_mult)
            if fired:
                exit_all, exit_reason = True, reason

        # 3. Time Stop (regime-scaled patience)
        if not exit_all:
            fired, reason = _check_time_stop(close, df, days_held, regime.time_days)
            if fired:
                exit_all, exit_reason = True, reason

        # 3b. Chandelier Exit (volatility-adaptive trail from the peak).
        #     Classic mult=3 in trending regimes, tightened in choppy ones.
        #     Anchored to the highest high since entry, so it lets winners run
        #     but cuts when price gives back too much of the move.
        if not exit_all:
            chand_mult = CHANDELIER_MULT_TREND if regime.trend == "trending" else CHANDELIER_MULT_CHOPPY
            fired, reason = _check_chandelier_exit(
                close, df, state[ticker].get("entry_date", ""), atr, chand_mult
            )
            if fired:
                exit_all, exit_reason = True, reason

        # 4. Trend Exit (regime-chosen). In trending regimes, Supertrend(10,3)
        #    is the primary trend-flip signal (cleaner than the raw EMA cross);
        #    the 20-EMA cross is kept as a secondary confirmation. Either firing
        #    exits. Choppy regimes use RSI mean-reversion instead.
        if not exit_all:
            if regime.trend == "trending":
                fired, reason = _check_supertrend_exit(df)
                if not fired:
                    fired, reason = _check_trending_exit(close, df)
            else:
                fired, reason = _check_choppy_exit(close, df, state[ticker])
            if fired:
                exit_all, exit_reason = True, reason

        # Execute full exit
        if exit_all:
            sell_all_qty = min(qty, qty_available)   # never exceed sellable shares
            reason_tag = exit_reason.split(":")[0].strip().replace(" ", "_")[:24]
            logger.info(
                f"ExitMgr | {ticker} SELL ALL {sell_all_qty} {'units' if is_crypto else 'shares'}  "
                f"[{regime.name}]  {exit_reason}"
            )
            result = (crypto_sell(ticker, sell_all_qty, reason=reason_tag) if is_crypto
                      else market_sell(ticker, sell_all_qty, reason=reason_tag))
            if result:
                _counter("exits_total", {
                    "ticker": ticker,
                    "reason": exit_reason.split(":")[0].strip(),
                    "regime": regime.name,
                })
                exits_placed += 1
                state.pop(ticker, None)
            continue

        # 5. Profit Ladder + adaptive trailing stop
        # Route to strategy-specific exit profile:
        #   institutional_breakout -> ATR-based 2/4 ATR ladder + Chandelier runner
        #   all others             -> existing percentage-based ladder
        strategy = state.get(ticker, {}).get("strategy", "momentum")

        if strategy == "institutional_breakout":
            # 5a-IB. Runner trail via tight Chandelier after both IB partials
            ts_now = state.get(ticker, {})
            if ts_now.get("ib_partial2_taken"):
                fired, reason = _check_chandelier_exit(
                    close, df, ts_now.get("entry_date", ""), atr, IB_TRAIL_MULT
                )
                if fired:
                    sell_all_qty = min(qty, qty_available)
                    logger.info(
                        f"ExitMgr | {ticker} SELL IB-RUNNER {sell_all_qty} "
                        f"{'units' if is_crypto else 'shares'}  [{regime.name}]  "
                        f"IB-Chandelier({IB_TRAIL_MULT}x): {reason}"
                    )
                    result = (crypto_sell(ticker, sell_all_qty, reason="ib_runner") if is_crypto
                              else market_sell(ticker, sell_all_qty, reason="ib_runner"))
                    if result:
                        _counter("exits_total", {"ticker": ticker, "reason": "ib_runner", "regime": regime.name})
                        exits_placed += 1
                        state.pop(ticker, None)
                    continue

            action, sell_qty, reason = _check_ib_profit_ladder(
                ticker, close, qty, state, atr, df
            )
        else:
            # 5a. Runner trail for percentage-based strategies
            ts_now = state.get(ticker, {})
            fired, reason = _check_chandelier_exit(
                close, df, ts_now.get("entry_date", ""), atr, CHANDELIER_MULT_RUNNER
            )
            if fired:
                sell_all_qty = min(qty, qty_available)
                logger.info(
                    f"ExitMgr | {ticker} SELL RUNNER {sell_all_qty} "
                    f"{'units' if is_crypto else 'shares'}  [{regime.name}]  "
                    f"runner Chandelier({CHANDELIER_MULT_RUNNER}x): {reason}"
                )
                result = (crypto_sell(ticker, sell_all_qty, reason="runner_chandelier") if is_crypto
                          else market_sell(ticker, sell_all_qty, reason="runner_chandelier"))
                if result:
                    _counter("exits_total", {"ticker": ticker,
                                             "reason": "runner_chandelier",
                                             "regime": regime.name})
                    exits_placed += 1
                    state.pop(ticker, None)
                continue

            action, sell_qty, reason = _check_profit_ladder(
                ticker, close, qty, state, effective_trail_pct
            )

        # ── Execute partial / trail for both strategy paths ──────────────────
        if action in ("partial", "partial2", "trail"):
            sell_qty = min(sell_qty, qty_available)   # never exceed sellable shares
            if sell_qty <= 0:
                logger.info(
                    f"ExitMgr | {ticker}: partial/trail wanted but 0 shares free "
                    f"(held for open orders) — skipping this cycle."
                )
                continue
            label = {
                "partial":  "SELL PARTIAL (rung 1)",
                "partial2": "SELL PARTIAL (rung 2)",
                "trail":    "SELL TRAIL",
            }[action]
            logger.info(
                f"ExitMgr | {ticker} {label} {sell_qty} {'units' if is_crypto else 'shares'}  "
                f"[{regime.name}]  {reason}"
            )
            result = (crypto_sell(ticker, sell_qty, reason=action) if is_crypto
                      else market_sell(ticker, sell_qty, reason=action))
            if result:
                _counter("exits_total", {
                    "ticker": ticker,
                    "reason": action,
                    "regime": regime.name,
                })
                exits_placed += 1
                if action == "trail":
                    state.pop(ticker, None)
        else:
            ts2 = state.get(ticker, {})
            phase = "post-rung2" if ts2.get("partial2_taken") else                     "post-rung1" if ts2.get("partial_taken")  else "pre-rung1"
            logger.debug(
                f"ExitMgr | {ticker} HOLD  "
                f"entry={entry:.2f}  close={close:.2f}  gain={gain_pct:+.2%}  "
                f"ATR={atr:.2f}  [{regime.name}]  days={days_held}  phase={phase}"
            )

    _save_state(state)
    return exits_placed
