"""
agents/market_regime.py -- Market regime classifier.

Classifies current market conditions along four dimensions:
  trend_regime  : strong_bull | bull | neutral | bear | strong_bear
  vol_regime    : low | normal | high | extreme
  market_type   : trending | mean_reverting | choppy
  risk_regime   : risk_on | risk_off

Then emits dynamic trading parameters:
  buy_threshold : total_score minimum to trigger BUY
  size_mult     : multiply base position size by this factor
  breadth_mult  : additional position size multiplier from breadth filter (0.25–1.0)

Data used:
  SPY  -- trend slope (20d linear regression), 50MA, 200MA, ADX(14)
  QQQ  -- tech leadership slope (20d linear regression)
  ^VIX -- implied volatility level
  Realized vol -- SPY 20-day annualised std dev
  ATR expansion -- ATR(14) vs 60-day ATR mean
  [RSP/XLK/XLU removed — caused Tiingo 429 noise; regime uses SPY+QQQ+VIX]

Breadth filter (4 binary conditions, 1 pt each):
  1. SPY above 50 EMA       -- intermediate trend intact
  2. QQQ 20-day slope > 0   -- tech / growth leadership
  3. VIX < 20               -- fear not elevated (replaces RSP breadth)
  4. VIX < 20               -- fear not elevated

  Score  breadth_mult   Interpretation
  ─────  ────────────   ─────────────────────────────
    4       1.00        All engines firing — full size
    3       0.75        Mostly healthy — slight reduction
    2       0.50        Mixed signals — half size
   0–1      0.25        Market internally weak — minimal exposure

Results are cached for 30 minutes so each scan run reuses the same regime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import ta
from loguru import logger

from data.market_data import get_ohlcv as _get_ohlcv

# ---------------------------------------------------------------------------
_CACHE_TTL_SEC = 1800   # refresh regime at most every 30 minutes
_cached_regime: Optional["RegimeState"] = None
_cache_time:    float = 0.0


@dataclass
class RegimeState:
    # Classification
    trend_regime: str   # strong_bull | bull | neutral | bear | strong_bear
    vol_regime:   str   # low | normal | high | extreme
    market_type:  str   # trending | mean_reverting | choppy
    risk_regime:  str   # risk_on | risk_off

    # Dynamic trading parameters
    buy_threshold: float   # total_score required to trigger BUY
    size_mult:     float   # scale base position size by this

    # Market breadth position-size multiplier (applied in execution layer on top of
    # quality tier mult — the two filters multiply together)
    breadth_mult:    float   # 1.00 / 0.75 / 0.50 / 0.25
    breadth_details: str     # one-line human-readable breakdown

    # Raw diagnostics (logged, stored in results)
    vix:             float
    adx:             float
    realized_vol:    float   # annualised %, 20-day
    atr_expansion:   float   # current ATR / 60-day avg ATR
    spy_slope_pct:   float   # 20-day linear regression slope as % of price
    spy_above_50ema: bool    # SPY close > 50-day EMA
    qqq_slope_pct:   float   # QQQ 20-day regression slope as % of price
    breadth_score:   float   # RSP/SPY slope proxy (-1 to +1 ish)

    # Fed funds context — OpenBB removed; fields hardcoded to None/unknown.
    # Additive diagnostic — does NOT alter buy_threshold / size_mult today.
    # Wire into regime_adjusted_thresholds yourself when you're ready to
    # have a tightening Fed bias thresholds upward / size downward.
    fed_funds_rate:  Optional[float]   # current effective rate, as a fraction
    fed_funds_label: str               # "easing" | "neutral" | "tightening" | "unknown"

    label:         str     # human-readable summary
    computed_at:   str     # ISO timestamp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV via the data layer (Tiingo primary, yfinance fallback).
    Translates Yahoo-style tickers to Tiingo equivalents where needed.
    """
    # Tiingo uses '^VIX' without the caret -- pass it directly
    tiingo_ticker = ticker.lstrip("^")
    df = _get_ohlcv(tiingo_ticker, period=period, interval="1d")
    df.columns = [c.lower() for c in df.columns]
    return df


def _slope_pct(series: pd.Series, window: int) -> float:
    """Linear regression slope over last `window` bars, as % of mean price."""
    y = series.iloc[-window:].values.astype(float)
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / float(np.mean(y)) * 100


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _classify_trend(slope: float, adx: float, above_200: bool) -> tuple[str, str]:
    """(trend_regime, market_type)"""
    if above_200 and slope > 0.15 and adx > 25:
        return "strong_bull", "trending"
    if above_200 and slope > 0.04:
        return "bull", "trending"
    if not above_200 and slope < -0.15 and adx > 25:
        return "strong_bear", "trending"
    if not above_200 and slope < -0.04:
        return "bear", "trending"
    if adx < 18:
        return "neutral", "mean_reverting" if slope > 0 else "choppy"
    return "neutral", "trending"


def _classify_vol(vix: float, rv: float, atr_exp: float) -> str:
    if vix > 35 or rv > 28 or atr_exp > 1.5:
        return "extreme"
    if vix > 22 or rv > 18 or atr_exp > 1.25:
        return "high"
    if vix < 13 and rv < 10:
        return "low"
    return "normal"


def _classify_risk(breadth: float, slope: float, tech_vs_util: float) -> str:
    # breadth and tech_vs_util removed from live data; use slope + QQQ slope proxy.
    # Called with breadth=0.0 and tech_vs_util=0.0 — use slope only.
    return "risk_on" if slope > 0 else "risk_off"


def _compute_breadth_mult(
    spy_above_50: bool,
    qqq_slope:    float,
    rsp_spy_slope: float,
    vix:           float,
) -> tuple[float, str]:
    """
    Score 4 binary breadth conditions and return (breadth_mult, details_string).

    Conditions (1 pt each):
      1. SPY above 50 EMA          -- intermediate uptrend intact
      2. QQQ 20-day slope > 0      -- tech/growth sector leading
      3. RSP/SPY slope > 0.05      -- equal-weight participation confirms
      4. VIX < 20                  -- fear/uncertainty not elevated

    breadth_mult table:
      Score 4 → 1.00  (all engines firing)
      Score 3 → 0.75  (mostly healthy)
      Score 2 → 0.50  (mixed signals, reduce size)
      Score 0-1 → 0.25 (internals weak, minimal exposure)
    """
    # 3-condition breadth model (RSP removed — Tiingo quota noise).
    c1 = spy_above_50        # SPY above 50 EMA
    c2 = qqq_slope > 0       # Tech/growth leading
    c3 = vix < 20            # Fear not elevated

    score = sum([c1, c2, c3])

    mult_map = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.25}
    mult = mult_map[score]

    details = (
        f"breadth={score}/3  "
        f"SPY>50EMA={'Y' if c1 else 'N'}  "
        f"QQQ_slope={'Y' if c2 else 'N'}({qqq_slope:+.3f}%)  "
        f"VIX<20={'Y' if c3 else 'N'}({vix:.1f})  "
        f"=> size_mult={mult:.2f}x"
    )

    return mult, details


def _regime_params(trend: str, vol: str, risk: str) -> tuple[float, float, str]:
    """Returns (buy_threshold, size_mult, label)."""
    # Base threshold from trend
    base_thresh = {
        "strong_bull": 2.5,
        "bull":        3.0,
        "neutral":     4.0,
        "bear":        5.0,
        "strong_bear": 5.5,
    }.get(trend, 3.5)
    label = {
        "strong_bull": "Strong Bull",
        "bull":        "Bull",
        "neutral":     "Neutral",
        "bear":        "Bear",
        "strong_bear": "Strong Bear",
    }.get(trend, "Unknown")

    # Volatility overlay
    vol_delta_thresh = {"low": -0.5, "normal": 0.0, "high": 1.0, "extreme": 1.5}
    vol_size         = {"low":  1.25, "normal": 1.0,  "high": 0.5,  "extreme": 0.25}
    vol_suffix       = {"low": " + Low Vol", "normal": "", "high": " + High Vol", "extreme": " + Extreme Vol"}

    thresh = max(1.5, base_thresh + vol_delta_thresh.get(vol, 0.0))
    size   = vol_size.get(vol, 1.0)
    label += vol_suffix.get(vol, "")

    # Risk-off penalty
    if risk == "risk_off":
        thresh += 0.5
        size   *= 0.75
        label  += " [Risk-Off]"

    return round(thresh, 2), round(min(size, 2.0), 3), label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_regime(force: bool = False) -> RegimeState:
    """Return current RegimeState, using 30-minute cache unless force=True."""
    global _cached_regime, _cache_time

    now = time.monotonic()
    if not force and _cached_regime and (now - _cache_time) < _CACHE_TTL_SEC:
        return _cached_regime

    logger.info("Computing market regime...")
    try:
        # SPY
        spy = _fetch("SPY", "1y")
        spy_c, spy_h, spy_l = spy["close"], spy["high"], spy["low"]

        slope     = _slope_pct(spy_c, 20)
        above_200 = bool(spy_c.iloc[-1] > spy_c.rolling(200).mean().iloc[-1])
        above_50  = bool(spy_c.iloc[-1] > spy_c.ewm(span=50, adjust=False).mean().iloc[-1])
        adx       = float(ta.trend.ADXIndicator(spy_h, spy_l, spy_c, window=14)
                          .adx().iloc[-1])
        rv        = float(spy_c.pct_change().dropna().tail(20).std() * (252 ** 0.5) * 100)
        atr_s     = ta.volatility.AverageTrueRange(spy_h, spy_l, spy_c, window=14) \
                      .average_true_range()
        atr_exp   = float(atr_s.iloc[-1] / atr_s.tail(60).mean())

        # VIX: Tiingo doesn't serve index data for ^VIX.
        # Try yfinance directly (bypassing Tiingo) using known-working symbols.
        # On failure default to 20 (neutral) so regime doesn't crash.
        vix_lvl = 20.0
        try:
            import yfinance as yf
            for _vix_sym in ("^VIX", "VIX", "VIXY"):
                try:
                    _vix_df = yf.download(_vix_sym, period="3mo", interval="1d",
                                          progress=False, timeout=10)
                    if _vix_df is not None and not _vix_df.empty:
                        _col = "Close" if "Close" in _vix_df.columns else _vix_df.columns[0]
                        vix_lvl = float(_vix_df[_col].squeeze().iloc[-1])
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # QQQ tech leadership slope
        qqq_slope = 0.0
        try:
            qqq_c     = _fetch("QQQ", "3mo")["close"]
            qqq_slope = _slope_pct(qqq_c, 20)
        except Exception:
            pass

        # Breadth: RSP/SPY removed (Tiingo rate-limit noise).
        # Replaced by QQQ slope + VIX in breadth_mult (3-condition model).
        breadth = 0.0

        # Sector rotation: XLK/XLU removed (Tiingo rate-limit noise).
        # Risk signal now derived from SPY slope + QQQ slope only.
        tech_vs_util = 0.0

        # Classify
        trend, mtype           = _classify_trend(slope, adx, above_200)
        vol                    = _classify_vol(vix_lvl, rv, atr_exp)
        risk                   = _classify_risk(breadth, slope, tech_vs_util)
        buy_thresh, size, lbl  = _regime_params(trend, vol, risk)

        # Breadth multiplier (applied additively in execution layer)
        b_mult, b_details = _compute_breadth_mult(above_50, qqq_slope, breadth, vix_lvl)

        # Fed funds context — OpenBB removed; hardcoded neutral defaults.
        # These fields remain in RegimeState for forward-compat but are not
        # used in trading logic.
        fed_rate_val: Optional[float] = None
        fed_label = "unknown" 

        regime = RegimeState(
            trend_regime     = trend,
            vol_regime       = vol,
            market_type      = mtype,
            risk_regime      = risk,
            buy_threshold    = buy_thresh,
            size_mult        = size,
            breadth_mult     = b_mult,
            breadth_details  = b_details,
            vix              = round(vix_lvl, 2),
            adx              = round(adx, 2),
            realized_vol     = round(rv, 2),
            atr_expansion    = round(atr_exp, 3),
            spy_slope_pct    = round(slope, 4),
            spy_above_50ema  = above_50,
            qqq_slope_pct    = round(qqq_slope, 4),
            breadth_score    = round(breadth, 4),
            fed_funds_rate   = round(fed_rate_val, 4) if fed_rate_val is not None else None,
            fed_funds_label  = fed_label,
            label            = lbl,
            computed_at      = datetime.now(timezone.utc).isoformat(),
        )
        ffr_part = "FFR=removed"
        logger.info(
            f"Regime: {regime.label}  |  "
            f"VIX={vix_lvl:.1f}  ADX={adx:.1f}  "
            f"RV={rv:.1f}%  ATR_exp={atr_exp:.2f}  "
            f"slope={slope:+.3f}%  breadth={breadth:+.3f}  "
            f"QQQ_slope={qqq_slope:+.3f}%  SPY>50EMA={above_50}  "
            f"{ffr_part}  "
            f"=> buy>={buy_thresh}  size={size}x"
        )
        logger.info(f"Breadth filter: {b_details}")
        _cached_regime = regime
        _cache_time    = now
        return regime

    except Exception as exc:
        logger.warning(f"Regime fetch failed ({exc}) -- using neutral defaults.")
        regime = RegimeState(
            trend_regime="neutral",  vol_regime="normal",
            market_type="choppy",    risk_regime="risk_on",
            buy_threshold=3.5,       size_mult=1.0,
            breadth_mult=0.5,        breadth_details="fetch failed -- defaulting to 0.5x",
            vix=0.0, adx=0.0, realized_vol=0.0,
            atr_expansion=1.0, spy_slope_pct=0.0,
            spy_above_50ema=False,   qqq_slope_pct=0.0,
            breadth_score=0.0,
            fed_funds_rate=None,     fed_funds_label="unknown",
            label="Unknown (fetch failed)",
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        _cached_regime = regime
        _cache_time    = now
        return regime



# ---------------------------------------------------------------------------
# Regime-adjusted threshold helper (used by screeners)
# ---------------------------------------------------------------------------

def regime_adjusted_thresholds(regime):
    """
    Map a RegimeState to dynamic scoring thresholds for both the momentum
    screener (0-100 score scale) and the unified screener (0.0-1.0 confidence).

    Returns a dict with keys:
      score_strong_buy  : int    momentum screener -- STRONG BUY floor
      score_buy         : int    momentum screener -- BUY floor
      score_watch       : int    momentum screener -- WATCH floor
      conf_strong_buy   : float  unified screener  -- STRONG BUY confidence
      conf_confirmed    : float  unified screener  -- CONFIRMED confidence
      conf_watch        : float  unified screener  -- WATCH confidence
      size_cap          : float  hard cap on position size_mult (0.25-1.00)
      veto_longs        : bool   True in strong_bear or bear+extreme vol
      regime_label      : str    human-readable summary for logging
    """
    trend = regime.trend_regime
    vol   = regime.vol_regime
    risk  = regime.risk_regime

    # Base score thresholds (0-100 scale) by trend regime.
    # Calibrated for the STRIPPED 3-factor momentum model (ema9_gt_21 + rel_vol
    # + gap, max 31 raw -> renormalised to 100). Natural peak in this model is
    # 65-75 for true breakouts; tier ratios match momentum_screener.SCORE_*
    # defaults. Old thresholds (95/90/80 ... 80/65/50) were tuned for the full
    # 11-factor model and rejected every ticker after the strip-down.
    SCORE_BASE = {
        "strong_bull": dict(strong_buy=55, buy=42, watch=28, size_cap=1.00),
        "bull":        dict(strong_buy=60, buy=45, watch=32, size_cap=1.00),
        "neutral":     dict(strong_buy=63, buy=48, watch=35, size_cap=0.75),
        "bear":        dict(strong_buy=70, buy=58, watch=45, size_cap=0.50),
        "strong_bear": dict(strong_buy=78, buy=68, watch=58, size_cap=0.25),
    }
    base = dict(SCORE_BASE.get(trend, SCORE_BASE["neutral"]))

    # Base confidence thresholds (0.0-1.0) by trend regime
    CONF_BASE = {
        "strong_bull": dict(conf_strong_buy=0.75, conf_confirmed=0.62, conf_watch=0.48),
        "bull":        dict(conf_strong_buy=0.80, conf_confirmed=0.68, conf_watch=0.52),
        "neutral":     dict(conf_strong_buy=0.80, conf_confirmed=0.68, conf_watch=0.52),
        "bear":        dict(conf_strong_buy=0.86, conf_confirmed=0.75, conf_watch=0.60),
        "strong_bear": dict(conf_strong_buy=0.90, conf_confirmed=0.82, conf_watch=0.70),
    }
    conf = dict(CONF_BASE.get(trend, CONF_BASE["neutral"]))

    # Volatility overlay: high vol raises score bar and reduces size
    vol_score_delta = dict(low=-3, normal=0, high=5, extreme=10)
    vol_size_mult   = dict(low=1.0, normal=1.0, high=0.75, extreme=0.50)
    vol_conf_delta  = dict(low=-0.02, normal=0.0, high=0.04, extreme=0.07)

    sd = vol_score_delta.get(vol, 0)
    cd = vol_conf_delta.get(vol, 0.0)
    vm = vol_size_mult.get(vol, 1.0)

    base["strong_buy"] = min(98, base["strong_buy"] + sd)
    base["buy"]        = min(95, base["buy"]        + sd)
    base["watch"]      = min(90, base["watch"]      + sd)
    base["size_cap"]   = round(base["size_cap"] * vm, 3)

    conf["conf_strong_buy"] = round(min(0.97, conf["conf_strong_buy"] + cd), 3)
    conf["conf_confirmed"]  = round(min(0.95, conf["conf_confirmed"]  + cd), 3)
    conf["conf_watch"]      = round(min(0.90, conf["conf_watch"]      + cd), 3)

    # Risk-off overlay: tighten further
    if risk == "risk_off":
        base["strong_buy"] = min(98, base["strong_buy"] + 3)
        base["buy"]        = min(95, base["buy"]        + 3)
        base["watch"]      = min(90, base["watch"]      + 3)
        base["size_cap"]   = round(base["size_cap"] * 0.80, 3)

        conf["conf_strong_buy"] = round(min(0.97, conf["conf_strong_buy"] + 0.03), 3)
        conf["conf_confirmed"]  = round(min(0.95, conf["conf_confirmed"]  + 0.03), 3)
        conf["conf_watch"]      = round(min(0.90, conf["conf_watch"]      + 0.03), 3)

    # Veto longs: strong_bear always; bear + extreme vol also
    veto_longs = (
        trend == "strong_bear"
        or (trend == "bear" and vol == "extreme")
    )

    return dict(
        score_strong_buy = base["strong_buy"],
        score_buy        = base["buy"],
        score_watch      = base["watch"],
        conf_strong_buy  = conf["conf_strong_buy"],
        conf_confirmed   = conf["conf_confirmed"],
        conf_watch       = conf["conf_watch"],
        size_cap         = base["size_cap"],
        veto_longs       = veto_longs,
        regime_label     = regime.label,
    )
