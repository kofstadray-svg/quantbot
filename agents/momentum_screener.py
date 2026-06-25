"""
agents/momentum_screener.py -- Momentum / breakout screener.

SCORING MODEL (weighted probabilistic, 0-100 pts):

  Factor groups and max points (total 100):
    1. Internal Strength  (23 pts): EMA9>21 (9), RSI zone (12), near HOD (2)
    2. Volume Expansion   (20 pts): Rel-vol graduated (12), abs vol (2), float rotation (6/3)
    3. Breakout Confirm   (16 pts): Gap graduated (10), entry timing setups (6)
    4. Daily VWAP          (6 pts): Price vs intraday VWAP (6)
    5. Volatility / Float  (5 pts): ATR% (5), small float (2)
    6. Relative Strength  (16 pts): excess return vs SPY (12), RS-ratio slope (4)
       -- distinguishes true leaders from stocks merely riding market beta.
    7. Anchored VWAP      (12 pts): gap-day anchor (above 4 + rising 3),
       52-week-breakout anchor (above 3 + rising 2). Institutional cost basis
       since the move began -- price above a RISING AVWAP = accumulation.
    8. Supertrend(10,3)    (8 pts): price above the ATR trend line (+ fresh-flip bonus).
    9. Volume Profile      (8 pts): breakout validation via HVN/LVN -- above the
       Point of Control (3) + clear overhead / no HVN wall within 3% (3) +
       sitting in a Low Volume Node / clear air (2). Recent ~60-day profile.

  All continuous metrics (rel-vol, gap, RSI) use graduated partial-credit scoring.

SIGNAL TIERS (default thresholds, overridden by regime when regime is passed):
  STRONG BUY  score >= 85  ->  size_mult 1.00
  BUY         score >= 70  ->  size_mult 0.75
  WATCH       score >= 55  ->  size_mult 0.50
  SKIP        score <  55  ->  size_mult 0.00

Entry Timing Filter (ORB / VWAP pullback / Bull Flag):
  Used as a score bonus (up to 10 pts in Breakout Confirmation group)
  AND as the final gate for auto-trade: STRONG BUY requires at least one.

Regime-Aware Thresholds:
  When a RegimeState is passed to screen(), thresholds and size_cap are
  dynamically adjusted via market_regime.regime_adjusted_thresholds().
  strong_bull lowers the bar; bear/strong_bear raise it (or veto longs).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math
import ta
from utils.indicators import supertrend_state
from loguru import logger
import time as _time

from data.market_data import get_ohlcv as _get_ohlcv, get_info as _get_info

# -- Benchmark (SPY) cache for Relative Strength -------------------------------
# Fetched once per scan and reused across all tickers (177+ symbols), with a
# short TTL so a long scan doesn't refetch. RS compares each stock to the market
# so we can tell genuine leaders from names merely riding beta.
_BENCH_CACHE: dict[str, tuple[float, "object"]] = {}
_BENCH_TTL_SEC = 600  # 10 min

def _get_benchmark_close(symbol: str = "SPY"):
    """Return the benchmark's 20d daily close series, cached for the scan."""
    now = _time.time()
    hit = _BENCH_CACHE.get(symbol)
    if hit and (now - hit[0]) < _BENCH_TTL_SEC:
        return hit[1]
    try:
        df = _get_ohlcv(symbol, period="20d", interval="1d")
        series = df["close"].squeeze() if df is not None and len(df) >= 6 else None
    except Exception as e:
        logger.warning(f"momentum_screener | benchmark {symbol} fetch failed: {e}")
        series = None
    _BENCH_CACHE[symbol] = (now, series)
    return series

def _relative_strength(stock_close, bench_close) -> dict:
    """
    Relative strength of a stock vs the benchmark over ~20 sessions.

    Returns dict with:
      excess_20d : stock_return_20d - bench_return_20d   (the headline RS number)
      ratio_slope: % change of (stock/bench) ratio over the window, annualised
                   per-bar -> rising ratio = persistent outperformance
      outperforming: bool (excess_20d > 0)
    Robust to NaN / short / misaligned series; returns zeros if uncomputable.
    """
    out = {"excess_20d": 0.0, "ratio_slope": 0.0, "outperforming": False}
    try:
        if stock_close is None or bench_close is None:
            return out
        s = stock_close.dropna(); b = bench_close.dropna()
        n = min(len(s), len(b))
        if n < 6:
            return out
        s = s.iloc[-n:]; b = b.iloc[-n:]
        s0, s1 = float(s.iloc[0]), float(s.iloc[-1])
        b0, b1 = float(b.iloc[0]), float(b.iloc[-1])
        if s0 <= 0 or b0 <= 0:
            return out
        stock_ret = (s1 - s0) / s0 * 100.0
        bench_ret = (b1 - b0) / b0 * 100.0
        excess = stock_ret - bench_ret
        # RS ratio slope: linear fit of (stock/bench) over the window, normalised
        ratio = (s.values / b.values)
        if ratio[0] != 0:
            import numpy as _np
            x = _np.arange(len(ratio), dtype=float)
            slope = _np.polyfit(x, ratio, 1)[0]          # ratio units per bar
            ratio_slope = slope / ratio[0] * 100.0       # % of starting ratio per bar
        else:
            ratio_slope = 0.0
        if math.isnan(excess): excess = 0.0
        if math.isnan(ratio_slope): ratio_slope = 0.0
        out.update(excess_20d=round(excess, 2),
                   ratio_slope=round(ratio_slope, 4),
                   outperforming=excess > 0)
    except Exception as e:
        logger.debug(f"momentum_screener | RS calc error: {e}")
    return out


# RegimeState is imported lazily inside screen() to avoid circular imports
# (market_regime has no dependency on momentum_screener)


# -- Score thresholds -----------------------------------------------------------

# Tier thresholds. These were originally tuned for the FULL 11-factor scoring
# (every factor contributing -> realistic peak scores 80-90). After stripping
# to the 3 walk-forward-robust factors (ema9_gt_21 + rel_vol + gap, max 31 raw
# renormalised to 100), the natural distribution shifted: a clean trending
# stock with a small gap typically scores 30-45, only true low-float breakouts
# (gap 8%+ AND rel_vol 2x+) push past 60. The old 85/70/55 thresholds rejected
# every ticker -- 65/65 scans in production logged "no advance to Stage 2".
# Recalibrated to match the stripped-model distribution while keeping the same
# tier ratio (~1.33x between WATCH/BUY/STRONG_BUY). Regime-adjusted equivalents
# in market_regime.regime_adjusted_thresholds() use the same scale.
SCORE_STRONG_BUY = 60   # true breakout (gap >= 8% + trend + some rel-vol)
SCORE_BUY        = 40   # solid setup (modest gap + trend)
SCORE_WATCH      = 28   # trending stock with at least one confirming factor

SIZE_MULT = {
    "STRONG BUY": 1.00,
    "BUY":        0.75,
    "WATCH":      0.50,
    "SKIP":       0.00,
}

# -- Scoring mode --------------------------------------------------------------
# Walk-forward + univariate factor testing (see backtesting/factor_backtest.py)
# showed the full 11-factor scoring OVERFITS: it loses money out-of-sample
# (-5.3% mean, degradation -0.47). Stripping to the 3 economically-distinct
# factors that each carried positive out-of-sample edge -- trend (EMA9>21),
# volume (rel-vol), breakout (gap) -- flipped the strategy to a small but REAL
# positive OOS edge (+1.2% mean, ~48% win, degradation ~+0.04).
#
# "stripped"  : score only the 3 robust factors, renormalised to 0-100. (default)
# "full"      : original 11-factor scoring (kept for comparison / A-B testing).
#
# The other factors (RS, AVWAP, Supertrend, Volume Profile) are STILL COMPUTED
# and surfaced in each result dict -- they remain valuable as exit-engine signals
# and dashboard context; they just no longer drive the entry score.
SCORING_MODE = "stripped"

# Factors that survived walk-forward, with their max points (used in stripped mode).
STRIPPED_FACTORS = {"ema9_gt_21", "rel_vol", "gap"}


# -- VWAP helper ---------------------------------------------------------------

def _calc_vwap(intraday) -> float | None:
    try:
        hi  = intraday["High"].squeeze()
        lo  = intraday["Low"].squeeze()
        cl  = intraday["Close"].squeeze()
        vol = intraday["Volume"].squeeze()
        tp  = (hi + lo + cl) / 3
        vwap = (tp * vol).cumsum() / vol.cumsum()
        return float(vwap.iloc[-1])
    except Exception:
        return None


# -- Anchored VWAP -------------------------------------------------------------
# Daily VWAP resets every session; anchored VWAP starts from a meaningful EVENT
# and accumulates forward. It answers: "what's the average price paid by everyone
# who got in since the move began?" Institutions defend these levels. Two anchors
# are detected from price alone (no earnings-API cost):
#   - gap day:        the most recent significant up-gap that kicked off the move
#   - 52w breakout:   the day price first broke to a 52-week high in the window
# Signals scored: price > AVWAP (holding institutional cost basis) and AVWAP
# rising (sustained accumulation, not distribution).

def _anchored_vwap_series(daily, anchor_idx: int):
    """Cumulative VWAP from anchor_idx forward. Returns the AVWAP pandas Series
    (indexed like daily from the anchor on) or None."""
    try:
        if anchor_idx is None or anchor_idx < 0 or anchor_idx >= len(daily) - 1:
            return None
        seg = daily.iloc[anchor_idx:]
        hi  = seg["High"].squeeze()
        lo  = seg["Low"].squeeze()
        cl  = seg["Close"].squeeze()
        vol = seg["Volume"].squeeze()
        tp  = (hi + lo + cl) / 3.0
        cum_vol = vol.cumsum()
        avwap = (tp * vol).cumsum() / cum_vol.replace(0, float("nan"))
        return avwap.dropna()
    except Exception:
        return None


def _find_gap_anchor(daily, min_gap_pct: float = 3.0, lookback: int = 120) -> int | None:
    """Index of the most recent significant up-gap (open vs prior close) within
    the lookback window. This marks where the current leg likely began."""
    try:
        op = daily["Open"].squeeze()
        cl = daily["Close"].squeeze()
        n  = len(daily)
        start = max(1, n - lookback)
        best = None
        for i in range(start, n):
            prev_close = float(cl.iloc[i - 1])
            today_open = float(op.iloc[i])
            if prev_close <= 0 or math.isnan(today_open):
                continue
            gap = (today_open - prev_close) / prev_close * 100.0
            if gap >= min_gap_pct:
                best = i   # keep the most RECENT qualifying gap
        return best
    except Exception:
        return None


def _find_52w_breakout_anchor(daily, lookback: int = 252) -> int | None:
    """Index of the day price first closed at a new (rolling) high after a base,
    within the window -- i.e. the 52-week-breakout anchor."""
    try:
        cl = daily["Close"].squeeze()
        n  = len(daily)
        if n < 30:
            return None
        win = min(lookback, n)
        seg = cl.iloc[-win:]
        running_max = seg.cummax()
        # breakout bar = first bar where close == new running max AND it exceeded
        # the prior max by a hair (a genuine new high, not the very first bar)
        best = None
        prior_max = float(seg.iloc[0])
        for k in range(1, len(seg)):
            c = float(seg.iloc[k])
            if c > prior_max * 1.001:      # new high (0.1% buffer vs noise)
                best = (n - win) + k       # keep most recent breakout
            prior_max = max(prior_max, c)
        return best
    except Exception:
        return None


def _avwap_signals(daily) -> dict:
    """Compute anchored-VWAP signals for gap and 52w-breakout anchors.
    Returns dict of booleans/values used by scoring + logging."""
    out = {
        "gap_anchor_date":   None, "above_gap_avwap":   None, "gap_avwap_rising":   None,
        "br52_anchor_date":  None, "above_br52_avwap":  None, "br52_avwap_rising":  None,
        "gap_avwap": None, "br52_avwap": None,
    }
    try:
        close = daily["Close"].squeeze()
        last  = float(close.iloc[-1])

        for name, idx in (("gap", _find_gap_anchor(daily)),
                          ("br52", _find_52w_breakout_anchor(daily))):
            if idx is None:
                continue
            av = _anchored_vwap_series(daily, idx)
            if av is None or len(av) < 3:
                continue
            cur_av = float(av.iloc[-1])
            # rising = AVWAP higher than ~5 bars ago (sustained accumulation)
            look = min(5, len(av) - 1)
            rising = float(av.iloc[-1]) > float(av.iloc[-1 - look])
            try:
                anchor_date = str(daily.index[idx].date())
            except Exception:
                anchor_date = None
            if name == "gap":
                out["gap_anchor_date"] = anchor_date
                out["above_gap_avwap"] = last > cur_av
                out["gap_avwap_rising"] = rising
                out["gap_avwap"] = round(cur_av, 2)
            else:
                out["br52_anchor_date"] = anchor_date
                out["above_br52_avwap"] = last > cur_av
                out["br52_avwap_rising"] = rising
                out["br52_avwap"] = round(cur_av, 2)
    except Exception as e:
        logger.debug(f"momentum_screener | AVWAP calc error: {e}")
    return out


# -- Volume Profile (HVN / LVN) ------------------------------------------------
# Volume tells you HOW MUCH traded over time; volume profile tells you AT WHAT
# PRICES it traded. High Volume Nodes (HVN) are price levels of acceptance --
# walls of supply/demand institutions defend (support/resistance). Low Volume
# Nodes (LVN) are rejection zones price slices through quickly.
#
# Breakout validation logic:
#   - A breakout is HIGH QUALITY when price has cleared the Point of Control
#     (the biggest HVN) and sits in clear air (an LVN) with no immediate
#     overhead HVN -- supply is gone, runway is open to the next node.
#   - A breakout is SUSPECT when an overhead HVN looms just above -- price is
#     about to hit a supply wall.

def _volume_profile(daily, bins: int = 40, lookback: int = 60) -> dict:
    """
    Build a volume-by-price histogram from daily bars and locate HVN/LVN.

    Uses a RECENT window (~60 sessions = the current base/consolidation), not a
    full year. For breakout validation we care about the nodes in the zone price
    is actually fighting through now -- a 1-year profile buries current
    resistance under last year's base and makes "above POC" fire for everyone.

    Approximates intraday volume distribution by spreading each day's volume
    across its high-low range (uniform), a robust daily proxy for a true tick
    profile. Returns dict with:
      poc            : Point of Control price (highest-volume bin)
      above_poc      : bool, current price above the POC
      nearest_hvn_above / _below : nearest High Volume Node price on each side
      nearest_lvn_above / _below : nearest Low Volume Node price on each side
      dist_to_hvn_above_pct      : % to the next overhead HVN (resistance ahead)
      dist_to_lvn_above_pct      : % to the next overhead LVN (runway)
      in_lvn         : bool, current price sits in a low-volume (clear-air) zone
      clear_overhead : bool, no HVN within ~3% above price (open runway)
    """
    out = {
        "poc": None, "above_poc": None,
        "nearest_hvn_above": None, "nearest_hvn_below": None,
        "nearest_lvn_above": None,
        "dist_to_hvn_above_pct": None, "dist_to_lvn_above_pct": None,
        "in_lvn": None, "clear_overhead": None,
    }
    try:
        seg = daily.iloc[-lookback:] if len(daily) > lookback else daily
        hi  = seg["High"].squeeze().astype(float).values
        lo  = seg["Low"].squeeze().astype(float).values
        vol = seg["Volume"].squeeze().astype(float).values
        price = float(daily["Close"].squeeze().iloc[-1])

        pmin, pmax = float(np.nanmin(lo)), float(np.nanmax(hi))
        if not (pmax > pmin) or math.isnan(price):
            return out

        edges = np.linspace(pmin, pmax, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        vp = np.zeros(bins)

        # Spread each day's volume across the bins its range covers (uniform).
        for h, l, v in zip(hi, lo, vol):
            if math.isnan(h) or math.isnan(l) or math.isnan(v) or v <= 0 or h < l:
                continue
            lo_b = np.searchsorted(edges, l, side="right") - 1
            hi_b = np.searchsorted(edges, h, side="right") - 1
            lo_b = max(0, min(bins - 1, lo_b))
            hi_b = max(0, min(bins - 1, hi_b))
            span = hi_b - lo_b + 1
            vp[lo_b:hi_b + 1] += v / span

        if vp.sum() <= 0:
            return out

        poc_idx = int(np.argmax(vp))
        poc = float(centers[poc_idx])

        # HVN = bins in the top 30% of volume; LVN = bottom 30%.
        hi_thresh = np.quantile(vp, 0.70)
        lo_thresh = np.quantile(vp, 0.30)
        hvn_prices = centers[vp >= hi_thresh]
        lvn_prices = centers[vp <= lo_thresh]

        def nearest(arr, above: bool):
            cand = arr[arr > price] if above else arr[arr < price]
            if len(cand) == 0:
                return None
            return float(cand.min() if above else cand.max())

        hvn_above = nearest(hvn_prices, True)
        hvn_below = nearest(hvn_prices, False)
        lvn_above = nearest(lvn_prices, True)

        # Which bin is current price in, and is it a low-volume bin?
        cur_b = max(0, min(bins - 1, np.searchsorted(edges, price, side="right") - 1))
        in_lvn = bool(vp[cur_b] <= lo_thresh)

        d_hvn_above = ((hvn_above - price) / price * 100.0) if hvn_above else None
        d_lvn_above = ((lvn_above - price) / price * 100.0) if lvn_above else None
        # Clear overhead = no HVN within 3% above current price.
        clear_overhead = (hvn_above is None) or (d_hvn_above is not None and d_hvn_above > 3.0)

        out.update(
            poc=round(poc, 2),
            above_poc=price > poc,
            nearest_hvn_above=round(hvn_above, 2) if hvn_above else None,
            nearest_hvn_below=round(hvn_below, 2) if hvn_below else None,
            nearest_lvn_above=round(lvn_above, 2) if lvn_above else None,
            dist_to_hvn_above_pct=round(d_hvn_above, 2) if d_hvn_above is not None else None,
            dist_to_lvn_above_pct=round(d_lvn_above, 2) if d_lvn_above is not None else None,
            in_lvn=in_lvn,
            clear_overhead=clear_overhead,
        )
    except Exception as e:
        logger.debug(f"momentum_screener | volume-profile error: {e}")
    return out


# -- Entry timing detectors (5-min intraday bars) ------------------------------

def _detect_orb(intraday) -> bool:
    """Opening Range Breakout -- price above the first-15-min high."""
    try:
        if len(intraday) < 4:
            return False
        or_high     = float(intraday["High"].squeeze().iloc[:3].max())
        current     = float(intraday["Close"].squeeze().iloc[-1])
        prev_closes = intraday["Close"].squeeze().iloc[3:-1]
        was_below   = any(float(c) < or_high for c in prev_closes)
        return was_below and current > or_high
    except Exception:
        return False


def _detect_vwap_pullback(intraday, vwap: float | None) -> bool:
    """
    First Pullback to VWAP -- within the last 8 bars price touched VWAP
    then bounced back above it.
    """
    try:
        if vwap is None or len(intraday) < 6:
            return False
        recent  = intraday.iloc[-8:]
        lows    = recent["Low"].squeeze()
        closes  = recent["Close"].squeeze()
        current = float(closes.iloc[-1])
        if current <= vwap:
            return False
        touched    = any(float(lo) <= vwap * 1.002 for lo in lows)
        recovering = current >= float(closes.max()) * 0.995
        return touched and recovering
    except Exception:
        return False


def _detect_bull_flag(intraday) -> bool:
    """
    Bull Flag Breakout -- sharp pole then tight consolidation,
    current bar breaks above the flag high.
    """
    try:
        if len(intraday) < 16:
            return False
        closes = intraday["Close"].squeeze()
        highs  = intraday["High"].squeeze()

        pole_bars   = closes.iloc[-14:-7]
        flag_closes = closes.iloc[-7:-1]
        flag_highs  = highs.iloc[-7:-1]

        pole_gain  = (float(pole_bars.iloc[-1]) - float(pole_bars.iloc[0])) / float(pole_bars.iloc[0])
        flag_high  = float(flag_highs.max())
        flag_low   = float(flag_closes.min())
        flag_mid   = (flag_high + flag_low) / 2 or 1.0
        flag_range = (flag_high - flag_low) / flag_mid
        current    = float(closes.iloc[-1])

        return pole_gain >= 0.02 and flag_range <= 0.015 and current > flag_high
    except Exception:
        return False


def _entry_setups(intraday, vwap: float | None) -> list[str]:
    """Return list of confirmed entry timing patterns (may be empty)."""
    if intraday is None or len(intraday) < 4:
        return []
    setups = []
    if _detect_orb(intraday):
        setups.append("ORB")
    if _detect_vwap_pullback(intraday, vwap):
        setups.append("VWAP")
    if _detect_bull_flag(intraday):
        setups.append("FLAG")
    return setups


# -- Weighted scoring engine ---------------------------------------------------

def _weighted_score(
    rel_vol:      float,
    gap_pct:      float,
    rsi:          float,
    today_vol:    float,
    float_shares: int | None,
    above_vwap:   bool | None,
    ema9_gt_21:   bool,
    near_hod:     bool | None,
    atr_pct:      float,
    entry_timing: list[str],
    rs_excess:    float = 0.0,
    rs_slope:     float = 0.0,
    avwap:        dict | None = None,
    st_bullish:   bool | None = None,
    st_flip_bull: bool = False,
    vprofile:     dict | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute weighted momentum score (0-100) with graduated partial credit.

    Returns
    -------
    total_score : float   0-100
    breakdown   : dict    per-factor points (for logging / UI)
    """
    pts: dict[str, float] = {}

    # 1. Relative Strength (35 pts) -------------------------------------------
    pts["ema9_gt_21"] = 4.0 if ema9_gt_21 else 0.0

    # RSI: graduated -- full credit in sweet spot, less near edges
    if   58 <= rsi <= 70:  pts["rsi"] = 12.0
    elif 55 <= rsi <  58:  pts["rsi"] =  8.0
    elif 70 <  rsi <= 72:  pts["rsi"] =  8.0   # edging exhaustion
    elif 50 <= rsi <  55:  pts["rsi"] =  4.0   # approaching but not there
    elif rsi > 72:         pts["rsi"] =  2.0   # late-stage exhaustion, minimal credit
    else:                  pts["rsi"] =  0.0

    if near_hod is True:   pts["near_hod"] = 1.0
    elif near_hod is None: pts["near_hod"] = 0.5   # unknown -> partial credit
    else:                  pts["near_hod"] = 0.0

    # 2. Volume Expansion (25 pts) --------------------------------------------
    # Relative volume: graduated -- reward extreme spikes more
    if   rel_vol >= 10.0:  pts["rel_vol"] = 12.0
    elif rel_vol >=  7.0:  pts["rel_vol"] = 10.0
    elif rel_vol >=  5.0:  pts["rel_vol"] =  8.0
    elif rel_vol >=  3.0:  pts["rel_vol"] =  5.0
    elif rel_vol >=  2.0:  pts["rel_vol"] =  2.0
    else:                  pts["rel_vol"] =  0.0

    pts["vol_2m"] = 2.0 if today_vol > 2_000_000 else 0.0

    if float_shares:
        pts["float_rotation"] = 6.0 if (today_vol / float_shares) > 0.50 else 0.0
    else:
        pts["float_rotation"] = 3.0   # half credit -- cannot verify

    # 3. Breakout Confirmation (20 pts) ---------------------------------------
    # Gap: graduated
    if   gap_pct >= 15.0:  pts["gap"] = 10.0
    elif gap_pct >= 10.0:  pts["gap"] =  9.0
    elif gap_pct >=  8.0:  pts["gap"] =  7.0
    elif gap_pct >=  5.0:  pts["gap"] =  4.0
    elif gap_pct >=  3.0:  pts["gap"] =  2.0
    else:                  pts["gap"] =  0.0

    # Entry timing: +3 pts per confirmed setup, capped at 10
    pts["entry_timing"] = min(2.0, len(entry_timing) * 1.0)

    # 4. VWAP Structure (12 pts) ----------------------------------------------
    if above_vwap is True:   pts["above_vwap"] = 2.0
    elif above_vwap is None: pts["above_vwap"] = 1.0   # intraday unavailable -> partial
    else:                    pts["above_vwap"] = 0.0

    # 5. Volatility / Float (8 pts) -------------------------------------------
    pts["atr_5pct"] = 3.0 if atr_pct > 5.0 else (2.0 if atr_pct > 3.0 else 0.0)
    if float_shares:
        pts["float_lt_30m"] = 2.0 if float_shares < 30_000_000 else 0.0
    else:
        pts["float_lt_30m"] = 1.0   # half credit -- unknown float

    # 6. Relative Strength vs market (18 pts) ---------------------------------
    # The single best leader/laggard discriminator: is the stock beating SPY?
    #   excess_20d = stock_return_20d - spy_return_20d   (headline outperformance)
    #   rs_slope   = slope of (stock/spy) ratio          (persistent leadership)
    # Excess return (up to 12 pts), graduated.
    if   rs_excess >=  8.0:  pts["rs_excess"] = 12.0
    elif rs_excess >=  4.0:  pts["rs_excess"] = 10.0
    elif rs_excess >=  2.0:  pts["rs_excess"] =  7.0
    elif rs_excess >=  0.0:  pts["rs_excess"] =  4.0   # merely keeping pace
    elif rs_excess >= -3.0:  pts["rs_excess"] =  1.0   # mild laggard
    else:                    pts["rs_excess"] =  0.0   # clear laggard
    # Ratio slope (up to 4 pts): rewards a *rising* RS line (trend of leadership).
    # Excess return is the stronger signal, so it carries more weight than slope.
    if   rs_slope >  0.10:   pts["rs_slope"] = 4.0
    elif rs_slope >  0.02:   pts["rs_slope"] = 3.0
    elif rs_slope >  0.0:    pts["rs_slope"] = 1.0
    else:                    pts["rs_slope"] = 0.0

    # 7. Anchored VWAP (12 pts) -----------------------------------------------
    # Price holding ABOVE a RISING anchored VWAP = institutions defending their
    # cost basis since the move began. Strongest when both anchors agree.
    av = avwap or {}
    def _av_pts(above, rising, w_above, w_rising):
        p = 0.0
        if above is True:   p += w_above
        if rising is True:  p += w_rising
        return p
    # Gap anchor (up to 7): the move's origin -- most actionable for momentum.
    pts["avwap_gap"]  = _av_pts(av.get("above_gap_avwap"),  av.get("gap_avwap_rising"),  4.0, 3.0)
    # 52w-breakout anchor (up to 5): longer-term institutional cost basis.
    pts["avwap_br52"] = _av_pts(av.get("above_br52_avwap"), av.get("br52_avwap_rising"), 3.0, 2.0)

    # 8. Supertrend(10,3) trend filter (8 pts) --------------------------------
    # Clean ATR-based trend read. Price above the line = uptrend. A fresh bullish
    # flip is the highest-quality entry timing (trend just turned up).
    if st_bullish is True:
        pts["supertrend"] = 8.0 if st_flip_bull else 6.0
    elif st_bullish is None:
        pts["supertrend"] = 3.0   # uncomputable -> partial credit
    else:
        pts["supertrend"] = 0.0   # below line = downtrend, no credit

    # 9. Volume Profile / HVN-LVN (8 pts) -------------------------------------
    # Breakout VALIDATION via where volume actually traded:
    #   - above Point of Control = cleared the biggest supply wall (institutions
    #     who accumulated there are now in profit / out of the way)
    #   - clear overhead (no HVN within ~3%) or sitting in an LVN = open runway,
    #     price can travel fast to the next node
    #   - an overhead HVN looming just above = suspect breakout (supply wall ahead)
    vp = vprofile or {}
    vp_pts = 0.0
    if vp.get("above_poc") is True:        vp_pts += 3.0   # cleared the main wall
    if vp.get("clear_overhead") is True:   vp_pts += 3.0   # open runway above
    elif vp.get("clear_overhead") is False: vp_pts += 0.0  # supply wall just above
    if vp.get("in_lvn") is True:           vp_pts += 2.0   # in clear air (rejection zone)
    pts["volume_profile"] = min(8.0, vp_pts)

    # -- Stripped scoring mode -------------------------------------------------
    # Keep only the 3 walk-forward-robust factors and renormalise so they fill
    # the full 0-100 range (otherwise their ~31-pt max could never clear the 55
    # WATCH threshold). The discarded factors stay in `pts` for logging but
    # contribute 0 to the total. EMA9>21 max is 4 in the full model; in stripped
    # mode we treat the 3 factors at their natural maxes (ema9=9-equivalent via
    # the trend signal, rel_vol=12, gap=10) -> renormalise by their kept max.
    if SCORING_MODE == "stripped":
        kept_max = {"ema9_gt_21": 9.0, "rel_vol": 12.0, "gap": 10.0}
        # ema9_gt_21 is scored at 4.0 in the full model; rescale to its 9.0 max
        kept_pts = {
            "ema9_gt_21": (9.0 if pts.get("ema9_gt_21", 0) > 0 else 0.0),
            "rel_vol":    pts.get("rel_vol", 0.0),
            "gap":        pts.get("gap", 0.0),
        }
        kept_sum = sum(kept_pts.values())
        kept_total_max = sum(kept_max.values())   # 31.0
        total = round(kept_sum / kept_total_max * 100.0, 1)
        # zero out non-kept factors in the breakdown for clarity
        for k in list(pts.keys()):
            if k not in STRIPPED_FACTORS:
                pts[k] = 0.0
        pts["ema9_gt_21"] = kept_pts["ema9_gt_21"]
        return total, pts

    total = sum(pts.values())
    return round(total, 1), pts


def _signal_from_score(
    score: float,
    thresh_strong_buy: int = SCORE_STRONG_BUY,
    thresh_buy:        int = SCORE_BUY,
    thresh_watch:      int = SCORE_WATCH,
) -> str:
    """Map weighted score to signal tier. Thresholds can be overridden by regime."""
    if score >= thresh_strong_buy: return "STRONG BUY"
    if score >= thresh_buy:        return "BUY"
    if score >= thresh_watch:      return "WATCH"
    return "SKIP"


# -- Per-ticker scorer ---------------------------------------------------------

def _score(ticker: str) -> dict | None:
    try:
        # Daily data: 1 year via data layer (Tiingo primary, yfinance fallback)
        daily = _get_ohlcv(ticker, period="1y", interval="1d")
        if daily is None or len(daily) < 6:
            return None

        # Normalise column names: Tiingo returns lowercase already; guard anyway
        daily.columns = [c.lower() for c in daily.columns]
        # Rename adj_close → close when using adjusted prices from Tiingo
        if "adj_close" in daily.columns and "close" not in daily.columns:
            daily = daily.rename(columns={"adj_close": "close"})

        close  = daily["close"].squeeze()
        high   = daily["high"].squeeze()
        low    = daily["low"].squeeze()
        open_  = daily["open"].squeeze()
        volume = daily["volume"].squeeze()

        # Determine whether the latest daily bar is COMPLETE. When the scan runs
        # pre-market (e.g. 05:50, hours before the 09:30 ET open) yfinance's
        # current-day bar has a NaN open and ~zero volume, which previously
        # produced Gap=+nan% and crushed rel_vol to ~0 for every ticker. If the
        # latest bar looks incomplete, shift the reference back one session so
        # gap and rel-vol are computed from the last COMPLETED bar instead.
        _last_open = float(open_.iloc[-1])
        _last_vol  = float(volume.iloc[-1])
        _med_vol   = float(volume.iloc[-6:-1].median()) or 1.0
        _bar_incomplete = (
            math.isnan(_last_open)
            or math.isnan(_last_vol)
            or _last_vol < 0.05 * _med_vol   # <5% of typical volume = pre-market stub
        )
        if _bar_incomplete:
            # Use the last completed session as "today".
            cur_idx, prev_idx = -2, -3
            logger.debug(
                f"momentum_screener | {ticker}: latest bar incomplete "
                f"(pre-market?) -- using prior completed session for gap/rel-vol."
            )
        else:
            cur_idx, prev_idx = -1, -2

        today_vol  = float(volume.iloc[cur_idx])
        avg_vol_5d = float(volume.iloc[cur_idx-5:cur_idx].mean()) or 1.0
        rel_vol    = today_vol / avg_vol_5d if avg_vol_5d > 0 else 0.0

        prev_close = float(close.iloc[prev_idx])
        today_open = float(open_.iloc[cur_idx])
        gap_pct    = ((today_open - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

        # Final NaN guard: never let a bad value poison scoring / thresholds.
        if math.isnan(gap_pct):  gap_pct = 0.0
        if math.isnan(rel_vol):  rel_vol = 0.0

        atr_val = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        atr_pct = float(atr_val) / float(close.iloc[-1]) * 100

        ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema9_gt_21 = ema9 > ema21

        rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])

        # Relative strength vs the market (SPY), cached once per scan.
        _bench = _get_benchmark_close("SPY")
        rs = _relative_strength(close, _bench)

        # Anchored VWAP signals (gap-day + 52-week-breakout anchors).
        avwap = _avwap_signals(daily)

        # Supertrend(10,3) on the daily series -- clean ATR-based trend filter.
        st = supertrend_state(high, low, close, period=10, multiplier=3.0)

        # Volume Profile (HVN/LVN) for breakout validation.
        vprofile = _volume_profile(daily)

        if rsi > 72:
            logger.warning(
                f"momentum_screener | {ticker} RSI={rsi:.1f} > 72 -- "
                f"exhaustion zone / likely trapped late buyers."
            )

        # Intraday data (VWAP + Near HOD + Entry Timing)
        # Tiingo IEX intraday via the data layer; yfinance fallback handles 5m bars
        intraday = _get_ohlcv(ticker, period="1d", interval="5m")
        vwap          = None
        current_price = float(close.iloc[-1])
        day_high      = float(high.iloc[-1])
        above_vwap    = None
        near_hod      = None
        entry_timing  = []

        if intraday is not None and len(intraday) >= 3:
            intraday.columns = [c.lower() for c in intraday.columns]
            vwap          = _calc_vwap(intraday)
            current_price = float(intraday["close"].squeeze().iloc[-1])
            day_high      = float(intraday["high"].squeeze().max())
            if vwap:
                above_vwap = current_price > vwap
            near_hod     = current_price >= day_high * 0.98
            entry_timing = _entry_setups(intraday, vwap)

        # Float (optional -- uses get_info which routes to yfinance .info dict)
        float_shares = None
        try:
            info = _get_info(ticker)
            fs = info.get("sharesOutstanding") or info.get("floatShares") or info.get("shares")
            if fs:
                float_shares = int(fs)
        except Exception:
            pass

        # Weighted score
        weighted_score, score_breakdown = _weighted_score(
            rel_vol      = rel_vol,
            gap_pct      = gap_pct,
            rsi          = rsi,
            today_vol    = today_vol,
            float_shares = float_shares,
            above_vwap   = above_vwap,
            ema9_gt_21   = ema9_gt_21,
            near_hod     = near_hod,
            atr_pct      = atr_pct,
            entry_timing = entry_timing,
            rs_excess    = rs["excess_20d"],
            rs_slope     = rs["ratio_slope"],
            avwap        = avwap,
            st_bullish   = st["bullish"],
            st_flip_bull = st["just_flipped"] and st["direction"] == 1,
            vprofile     = vprofile,
        )

        signal    = _signal_from_score(weighted_score)
        size_mult = SIZE_MULT[signal]

        # Legacy binary checks (kept for logging / downstream compat)
        checks: dict[str, bool | None] = {
            "float_lt_30m":         (float_shares < 30_000_000)        if float_shares else None,
            "rel_vol_5x":           rel_vol    > 5.0,
            "gap_8pct":             gap_pct    > 8.0,
            "vol_2m":               today_vol  > 2_000_000,
            "above_vwap":           above_vwap,
            "ema9_gt_21":           ema9_gt_21,
            "near_hod":             near_hod,
            "atr_5pct":             atr_pct    > 5.0,
            "float_rotation_50pct": (today_vol / float_shares > 0.5)   if float_shares else None,
            "rsi_55_72":            55 <= rsi <= 72,
        }
        known    = {k: v for k, v in checks.items() if v is not None}
        passed   = sum(1 for v in known.values() if v)
        failures = [k for k, v in known.items() if not v]

        # Breakdown detection (bearish put signal)
        breakdown = bool(
            rsi         < 35           and
            ema9        < ema21        and
            gap_pct     < -3.0         and
            rel_vol     > 2.0          and
            atr_pct     > 3.0          and
            (above_vwap is False or above_vwap is None)
        )

        return {
            # Core signal
            "ticker":          ticker,
            "signal":          signal,          # STRONG BUY / BUY / WATCH / SKIP
            "weighted_score":  weighted_score,  # 0-100
            "size_mult":       size_mult,        # 0.00 / 0.50 / 0.75 / 1.00
            "score_breakdown": score_breakdown,  # per-factor pts dict
            # Legacy counts (used in logging + backward compat)
            "passed":          passed,
            "total":           len(known),
            "failures":        failures,
            # Market data
            "rel_volume":      round(rel_vol, 2),
            "gap_pct":         round(gap_pct, 2),
            "atr_pct":         round(atr_pct, 2),
            "ema9":            round(ema9, 2),
            "ema21":           round(ema21, 2),
            "rsi":             round(rsi, 1),
            "price":           round(current_price, 2),
            "vwap":            round(vwap, 2) if vwap else None,
            "day_high":        round(day_high, 2),
            "float_shares":    float_shares,
            "float_rotation":  round(today_vol / float_shares, 2) if float_shares else None,
            "entry_timing":    entry_timing,
            "breakdown":       breakdown,
            # Relative strength vs SPY
            "rs_excess_20d":   rs["excess_20d"],
            "rs_ratio_slope":  rs["ratio_slope"],
            "rs_outperform":   rs["outperforming"],
            # Anchored VWAP
            "gap_avwap":         avwap["gap_avwap"],
            "above_gap_avwap":   avwap["above_gap_avwap"],
            "gap_avwap_rising":  avwap["gap_avwap_rising"],
            "gap_anchor_date":   avwap["gap_anchor_date"],
            "br52_avwap":        avwap["br52_avwap"],
            "above_br52_avwap":  avwap["above_br52_avwap"],
            "br52_avwap_rising": avwap["br52_avwap_rising"],
            "br52_anchor_date":  avwap["br52_anchor_date"],
            # Supertrend(10,3)
            "supertrend":        st["value"],
            "st_bullish":        st["bullish"],
            "st_direction":      st["direction"],
            "st_dist_pct":       st["dist_pct"],
            "st_just_flipped":   st["just_flipped"],
            # Volume Profile / HVN-LVN
            "vp_poc":            vprofile["poc"],
            "vp_above_poc":      vprofile["above_poc"],
            "vp_hvn_above":      vprofile["nearest_hvn_above"],
            "vp_hvn_below":      vprofile["nearest_hvn_below"],
            "vp_dist_to_hvn_above_pct": vprofile["dist_to_hvn_above_pct"],
            "vp_in_lvn":         vprofile["in_lvn"],
            "vp_clear_overhead": vprofile["clear_overhead"],
        }

    except Exception as e:
        logger.warning(f"momentum_screener | {ticker} error: {e}")
        return None


# -- Public screen() function --------------------------------------------------

def screen(watchlist: list[str], regime=None) -> list[dict]:
    """
    Screen a list of tickers. Returns results sorted by weighted_score (best first).

    Args:
        watchlist : list of ticker symbols
        regime    : optional RegimeState from market_regime.compute_regime().
                    When provided, score thresholds and size_cap are dynamically
                    adjusted to match current market conditions.
                    Pass None to use static module-level defaults.
    """
    # -- Resolve thresholds (regime-aware or static defaults) ------------------
    if regime is not None:
        from agents.market_regime import regime_adjusted_thresholds
        rt = regime_adjusted_thresholds(regime)
        thresh_sb   = rt["score_strong_buy"]
        thresh_buy  = rt["score_buy"]
        thresh_wch  = rt["score_watch"]
        size_cap    = rt["size_cap"]
        veto_longs  = rt["veto_longs"]
        logger.info(
            f"Momentum screener | regime={rt['regime_label']}  "
            f"thresholds=SB:{thresh_sb}/B:{thresh_buy}/W:{thresh_wch}  "
            f"size_cap={size_cap:.2f}  veto_longs={veto_longs}"
        )
    else:
        thresh_sb   = SCORE_STRONG_BUY
        thresh_buy  = SCORE_BUY
        thresh_wch  = SCORE_WATCH
        size_cap    = 1.00
        veto_longs  = False

    results = []
    for ticker in watchlist:
        logger.info(f"Momentum scan: {ticker}...")
        r = _score(ticker)
        if r is None:
            continue

        # Veto non-breakdown longs in bear regimes
        if veto_longs and not r.get("breakdown"):
            logger.info(f"Momentum | {ticker:<6} VETOED (bear regime, no breakdown signal)")
            continue

        # Re-map signal and size_mult using regime-adjusted thresholds
        signal = _signal_from_score(r["weighted_score"], thresh_sb, thresh_buy, thresh_wch)
        mult   = SIZE_MULT[signal]
        # Apply regime size cap on top of tier mult
        mult   = round(min(mult, size_cap), 3)

        r["signal"]    = signal
        r["size_mult"] = mult

        fails  = ", ".join(r["failures"])    if r["failures"]     else "--"
        setups = ", ".join(r["entry_timing"]) if r["entry_timing"] else "none"
        bd     = r["score_breakdown"]

        if SCORING_MODE == "stripped":
            # In stripped mode the old 11-factor binary checks (float_lt_30m,
            # rel_vol_5x, gap_8pct, rsi_55_72 ...) are meaningless -- a ticker
            # can be a valid BUY with rel_vol=0.5x and no 8%+ gap. Show only
            # the 3 factors that actually drive the stripped score, using the
            # thresholds the scoring engine rewards (not the legacy 5x/8% gates).
            sf = []
            if not (bd.get("ema9_gt_21", 0) > 0):
                sf.append("ema9_lt_21")
            if r.get("rel_volume", 0) < 2.0:
                sf.append("rel_vol_lt2x")
            if abs(r.get("gap_pct", 0)) < 3.0:
                sf.append("gap_lt3pct")
            fail_tag = ("FAIL: " + ", ".join(sf)) if sf else "ALL PASS"
            score_str = (
                f"EMA={'ok' if bd.get('ema9_gt_21',0) > 0 else 'no'}"
                f" Vol={r.get('rel_volume',0):.1f}x"
                f" Brk={bd.get('gap',0):.0f}pts"
            )
        else:
            fail_tag  = "FAIL: " + fails if r["failures"] else "ALL PASS"
            score_str = (
                f"RS={bd.get('ema9_gt_21',0)+bd.get('rsi',0)+bd.get('near_hod',0):.0f}"
                f" Vol={bd.get('rel_vol',0)+bd.get('vol_2m',0)+bd.get('float_rotation',0):.0f}"
                f" Brk={bd.get('gap',0)+bd.get('entry_timing',0):.0f}"
                f" VWAP={bd.get('above_vwap',0):.0f}"
                f" Vlt={bd.get('atr_5pct',0)+bd.get('float_lt_30m',0):.0f}"
            )
        logger.info(
            f"Momentum | {r['ticker']:<6} [{signal:10s}] "
            f"score={r['weighted_score']:5.1f}/100  x{mult:.2f}  "
            f"({score_str})  "
            f"RelVol={r['rel_volume']}x  Gap={r['gap_pct']:+.1f}%  "
            f"RSI={r['rsi']}  ATR={r['atr_pct']:.1f}%  "
            f"Entry=[{setups}]  "
            f"{fail_tag}"
        )
        results.append(r)

    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    return results
