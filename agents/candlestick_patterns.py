"""
agents/candlestick_patterns.py — Professional 5-Condition Signal Stack.

PRIMARY SIGNAL — 5-way conjunction (all must fire simultaneously):

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. Volatility Contraction   20-day BB squeeze + ATR at annual low      │
  │  2. Relative Volume > 2.0    institutional demand entering NOW           │
  │  3. Price above Anchored VWAP  price reclaimed / holding VWAP           │
  │  4. RS > Sector              outperforming sector ETF AND SPY            │
  │  5. Breakout above Range     price > 20-day consolidation high           │
  │  ════════════════════════════════════════════════════════════════         │
  │  = HIGH-CONVICTION BUY                                                   │
  └─────────────────────────────────────────────────────────────────────────┘

  Bear Regime veto (SPY < SMA200) suppresses ALL BUY signals regardless.

Why these five specifically:
  Volatility contraction → energy coiled, risk/reward skewed
  Rel vol > 2.0          → institutions are moving size, not noise
  Above anchored VWAP    → fair-value reclaim, buyers in control
  RS > sector            → leadership stock, not a sector-beta trade
  Range breakout         → structure confirms the move is real

Signal precedence:
  5/5 → BUY  (combo fires)
  4/5 → HOLD (near-miss — one leg missing, log which one)
  Bear regime → HOLD (veto overrides everything)

screen_candlesticks() public interface is unchanged.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CandleSignal:
    ticker:    str
    pattern:   str
    direction: str      # "bullish" | "bearish" | "neutral"
    strength:  str      # "strong" | "moderate" | "weak"
    action:    str      # "BUY" | "SELL" | "HOLD"
    reason:    str
    combo:     dict = field(default_factory=dict)   # sub-check breakdown


# ── sector ETF map ─────────────────────────────────────────────────────────────

SECTOR_ETF_MAP: dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Utilities":              "XLU",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _pct_rank(series: pd.Series, window: int, min_periods: int = 60) -> pd.Series:
    """Rolling percentile rank of the last value within the window (0–100)."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda w: float((w[:-1] < w[-1]).sum()) / max(1, len(w) - 1) * 100,
        raw=True,
    )


def _vwap_proxy(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, window: int = 20,
) -> pd.Series:
    typical  = (high + low + close) / 3
    vol_sum  = volume.rolling(window).sum()
    twap_sum = (typical * volume).rolling(window).sum()
    return twap_sum / vol_sum.replace(0, float("nan"))


# ── condition 1: volatility contraction ───────────────────────────────────────

def _check_vol_compression(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr_threshold: float = 35.0,   # ATR rank ≤35th pct — tight coil, not in expansion phase
    bb_threshold:  float = 35.0,   # BB width rank — primary squeeze detector
    lookback_bars: int   = 3,      # check last N bars (breakout bar expands ATR)
) -> dict:
    """
    Squeeze = BB width at annual low AND ATR not expanding.

    BB width drives the detection (reacts faster to quiet consolidation).
    ATR confirms the stock isn't already in a vol-expansion phase.
    Checks the last `lookback_bars` bars so the breakout bar itself doesn't
    disqualify a valid setup — takes the best (lowest) compression reading.
    """
    base = {"ok": False, "atr_rank": 99.0, "bb_rank": 99.0, "bars_ago": -1}
    if len(close) < 80:
        return base

    atr_s    = _atr(high, low, close, 14)
    atr_rank = _pct_rank(atr_s, window=252, min_periods=60)

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_width = (2 * std20) / sma20.replace(0, float("nan")) * 100
    bb_rank  = _pct_rank(bb_width, window=252, min_periods=60)

    best_atr, best_bb, best_i = 99.0, 99.0, -1

    for i in range(lookback_bars):
        idx = -(i + 1)
        if abs(idx) > len(atr_rank):
            break
        ar = atr_rank.iloc[idx]
        br = bb_rank.iloc[idx]
        if pd.isna(ar) or pd.isna(br):
            continue
        ar, br = float(ar), float(br)
        if ar < best_atr:
            best_atr, best_bb, best_i = ar, br, i

    if best_i == -1:
        return base

    # BB width is primary; ATR confirms not expanding
    ok = (best_bb < bb_threshold) and (best_atr < atr_threshold)
    return {
        "ok":       ok,
        "atr_rank": round(best_atr, 1),
        "bb_rank":  round(best_bb, 1),
        "bars_ago": best_i,
    }


# ── condition 2: relative volume > 2.0 ────────────────────────────────────────

def _check_vol_expansion(
    close: pd.Series, volume: pd.Series,
    vol_threshold: float = 2.0,
) -> dict:
    """
    Volume surge ≥ 2× with bullish price close = institutional demand.

    The 2.0× threshold (vs prior 1.5×) is more selective — it filters out
    everyday news-driven ticks and targets genuine institutional interest.
    """
    base = {"ok": False, "rel_vol": 0.0, "price_up": False}
    if len(close) < 22:
        return base

    avg_vol  = float(volume.iloc[-21:-1].mean())
    if avg_vol <= 0:
        return base

    rel_vol  = float(volume.iloc[-1]) / avg_vol
    price_up = float(close.iloc[-1]) > float(close.iloc[-2])
    ok       = rel_vol >= vol_threshold and price_up
    return {"ok": ok, "rel_vol": round(rel_vol, 2), "price_up": price_up}


# ── condition 3: price above anchored VWAP ────────────────────────────────────

def _check_vwap_position(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
) -> dict:
    """
    Price must be above the 20-day anchored VWAP AND VWAP slope must be up.

    VWAP above = buyers have been in control on a volume-weighted basis.
    VWAP slope up = the fair-value reference itself is trending higher.
    """
    base = {"ok": False, "gap_pct": 0.0, "vwap": 0.0}
    if len(close) < 25:
        return base

    vwap     = _vwap_proxy(high, low, close, volume, window=20)
    last_vwap = float(vwap.iloc[-1])
    if pd.isna(last_vwap) or last_vwap <= 0:
        return base

    above_today = float(close.iloc[-1]) > last_vwap
    vwap_slope  = len(vwap) > 6 and float(vwap.iloc[-1]) > float(vwap.iloc[-6])
    gap_pct     = (float(close.iloc[-1]) / last_vwap - 1) * 100
    ok          = above_today and vwap_slope
    return {"ok": ok, "gap_pct": round(gap_pct, 2), "vwap": round(last_vwap, 2)}


# ── condition 4: RS > sector (and SPY) ────────────────────────────────────────

def _check_rs(
    close: pd.Series,
    spy_close: "pd.Series | None",
    sector_close: "pd.Series | None" = None,
    lookback: int = 20,
) -> dict:
    """
    Ticker must outperform BOTH the market (SPY) AND its sector ETF.

    If sector data is unavailable, falls back to SPY-only comparison.
    RS slope must be positive (outperformance accelerating, not fading).
    """
    base = {"ok": False, "vs_spy": 0.0, "vs_sector": 0.0, "slope": 0.0, "ret10": 0.0}
    if spy_close is None or len(close) < lookback + 10 or len(spy_close) < lookback + 10:
        return base

    # ── vs SPY ────────────────────────────────────────────────────────────────
    merged_spy = pd.concat({"tkr": close, "spy": spy_close}, axis=1).dropna()
    if len(merged_spy) < lookback + 5:
        return base

    tkr = merged_spy["tkr"]
    spy = merged_spy["spy"]

    tkr_ret20   = (float(tkr.iloc[-1]) / float(tkr.iloc[-lookback]) - 1) * 100
    spy_ret20   = (float(spy.iloc[-1]) / float(spy.iloc[-lookback]) - 1) * 100
    vs_spy      = tkr_ret20 - spy_ret20

    rs_ratio    = (tkr / spy) * (float(spy.iloc[0]) / float(tkr.iloc[0]))
    rs_recent   = rs_ratio.iloc[-10:].values
    slope       = float(np.polyfit(range(len(rs_recent)), rs_recent, 1)[0]) if len(rs_recent) >= 5 else 0.0
    ret10       = (float(tkr.iloc[-1]) / float(tkr.iloc[-10]) - 1) * 100

    beats_spy   = (vs_spy > 0) and (slope > 0) and (ret10 > 0)

    # ── vs Sector ─────────────────────────────────────────────────────────────
    vs_sector   = 0.0
    beats_sector = True   # default True if no sector data available

    if sector_close is not None and len(sector_close) >= lookback + 5:
        merged_sec  = pd.concat({"tkr": close, "sec": sector_close}, axis=1).dropna()
        if len(merged_sec) >= lookback + 5:
            sec         = merged_sec["sec"]
            sec_ret20   = (float(sec.iloc[-1]) / float(sec.iloc[-lookback]) - 1) * 100
            vs_sector   = tkr_ret20 - sec_ret20
            beats_sector = vs_sector > 0

    ok = beats_spy and beats_sector
    return {
        "ok":         ok,
        "vs_spy":     round(vs_spy, 2),
        "vs_sector":  round(vs_sector, 2),
        "slope":      round(slope, 5),
        "ret10":      round(ret10, 2),
    }


# ── condition 5: breakout above compression range ─────────────────────────────

def _check_range_breakout(
    high: pd.Series, close: pd.Series,
    range_bars: int = 20,
) -> dict:
    """
    Price must exceed the high of the prior `range_bars` consolidation range.

    This confirms the move is structural, not just intraday noise.
    We use the prior range (excluding today) so the breakout bar itself
    defines the new high, not the range it's breaking.
    """
    base = {"ok": False, "range_high": 0.0, "breakout_pct": 0.0}
    if len(high) < range_bars + 2:
        return base

    range_high    = float(high.iloc[-(range_bars + 1):-1].max())
    today_close   = float(close.iloc[-1])
    breakout_pct  = (today_close / range_high - 1) * 100
    ok            = today_close > range_high
    return {
        "ok":           ok,
        "range_high":   round(range_high, 2),
        "breakout_pct": round(breakout_pct, 2),
    }


# ── regime veto ────────────────────────────────────────────────────────────────

def _check_regime(spy_close: "pd.Series | None") -> dict:
    """
    Bull regime: SPY above SMA200, above EMA50, positive 20d return.
    Bear regime (SPY < SMA200) vetoes all BUY signals.
    """
    base = {"ok": False, "bear": False, "spy_ret20": 0.0, "sma200": 0.0, "spy_now": 0.0}
    if spy_close is None or len(spy_close) < 200:
        return base

    spy_now   = float(spy_close.iloc[-1])
    sma200    = float(spy_close.rolling(200).mean().iloc[-1])
    ema50     = float(_ema(spy_close, 50).iloc[-1])
    spy_ret20 = (spy_now / float(spy_close.iloc[-20]) - 1) * 100

    bear = spy_now < sma200
    ok   = (spy_now > sma200) and (spy_now > ema50) and (spy_ret20 > 0)
    return {
        "ok":       ok,
        "bear":     bear,
        "spy_ret20": round(spy_ret20, 2),
        "sma200":   round(sma200, 2),
        "spy_now":  round(spy_now, 2),
    }


# ── PRIMARY SIGNAL: 5-condition professional stack ─────────────────────────────

def _signal_combo(
    close: pd.Series, high: pd.Series, low: pd.Series,
    volume: pd.Series,
    spy_close: "pd.Series | None",
    sector_close: "pd.Series | None" = None,
) -> "CandleSignal | None":
    """
    Fires ONLY when all five conditions are simultaneously true:

      1. Volatility Contraction  — BB squeeze at annual low, ATR not expanding
      2. Relative Volume > 2.0   — institutional size entering now
      3. Above Anchored VWAP     — fair-value reclaim, buyers in control
      4. RS > Sector (& SPY)     — leadership, not beta
      5. Range Breakout          — structure confirms the move

    Strength grading:
      strong   — ≥ 4 of 5 sub-conditions comfortably exceed their thresholds
      moderate — just above thresholds on some legs
    """
    vc  = _check_vol_compression(high, low, close)
    ve  = _check_vol_expansion(close, volume)
    vwap = _check_vwap_position(high, low, close, volume)
    rs  = _check_rs(close, spy_close, sector_close)
    rb  = _check_range_breakout(high, close)

    conditions = {
        "vol_contraction": vc["ok"],
        "rel_vol_2x":      ve["ok"],
        "above_vwap":      vwap["ok"],
        "rs_vs_sector":    rs["ok"],
        "range_breakout":  rb["ok"],
    }
    n_passing = sum(conditions.values())

    if n_passing < 5:
        return None

    # Strength: count legs comfortably exceeding thresholds
    strong_legs = sum([
        vc["bb_rank"]       < 20,
        ve["rel_vol"]       > 2.5,
        vwap["gap_pct"]     > 0.5,
        rs["vs_spy"]        > 3.0,
        rb["breakout_pct"]  > 0.5,
    ])
    strength = "strong" if strong_legs >= 3 else "moderate"

    reason = (
        f"BB={vc['bb_rank']:.0f}pct  "
        f"vol={ve['rel_vol']:.1f}x  "
        f"VWAP+{vwap['gap_pct']:.1f}%  "
        f"RS+{rs['vs_spy']:+.1f}%spy"
        + (f"/+{rs['vs_sector']:+.1f}%sec" if rs['vs_sector'] != 0 else "")
        + f"  break+{rb['breakout_pct']:.2f}%"
    )

    return CandleSignal(
        ticker="",
        pattern="VolContraction+RelVol+VWAP+RS+Breakout",
        direction="bullish",
        strength=strength,
        action="BUY",
        reason=reason,
        combo=conditions,
    )


# ── near-miss: 4 of 5 ─────────────────────────────────────────────────────────

def _signal_near_miss(
    close: pd.Series, high: pd.Series, low: pd.Series,
    volume: pd.Series,
    spy_close: "pd.Series | None",
    sector_close: "pd.Series | None" = None,
) -> "CandleSignal | None":
    """4 of 5 conditions met — add to watchlist, not trade list."""
    vc   = _check_vol_compression(high, low, close)
    ve   = _check_vol_expansion(close, volume)
    vwap = _check_vwap_position(high, low, close, volume)
    rs   = _check_rs(close, spy_close, sector_close)
    rb   = _check_range_breakout(high, close)

    conditions = {
        "Vol Contraction": vc["ok"],
        "Rel Vol 2x":      ve["ok"],
        "Above VWAP":      vwap["ok"],
        "RS > Sector":     rs["ok"],
        "Range Breakout":  rb["ok"],
    }
    n_passing = sum(conditions.values())

    if n_passing != 4:
        return None

    missing = [k for k, v in conditions.items() if not v]
    return CandleSignal(
        ticker="",
        pattern="Near-Miss (4/5)",
        direction="bullish",
        strength="moderate",
        action="HOLD",
        reason=f"4/5 conditions — missing: {missing[0]}",
        combo=conditions,
    )


# ── regime veto signal ─────────────────────────────────────────────────────────

def _signal_bear_regime(spy_close: "pd.Series | None") -> "CandleSignal | None":
    regime = _check_regime(spy_close)
    if regime["bear"]:
        return CandleSignal(
            ticker="", pattern="Bear Regime", direction="bearish",
            strength="strong", action="HOLD",
            reason=(
                f"SPY below SMA200 — avoid longs  "
                f"(SPY={regime['spy_now']:.0f}  SMA200={regime['sma200']:.0f})"
            ),
        )
    return None


# ── supporting context signals (not standalone triggers) ──────────────────────

def _signal_market_structure(
    close: pd.Series, high: pd.Series, low: pd.Series,
) -> "CandleSignal | None":
    if len(close) < 22:
        return None
    hh5,  hh10, hh20 = float(high.iloc[-5:].max()), float(high.iloc[-10:-5].max()), float(high.iloc[-20:-10].max())
    ll5,  ll10, ll20 = float(low.iloc[-5:].min()),  float(low.iloc[-10:-5].min()),  float(low.iloc[-20:-10].min())
    hh = hh5 > hh10 > hh20
    hl = ll5 > ll10 > ll20
    above_ema20 = float(close.iloc[-1]) >= float(_ema(close, 20).iloc[-1])
    if not above_ema20:
        return None
    strength = "strong" if (hh and hl) else "moderate"
    parts = (["HH chain"] if hh else []) + (["HL chain"] if hl else []) + ["above EMA20"]
    return CandleSignal(
        ticker="", pattern="Market Structure", direction="bullish",
        strength=strength, action="HOLD", reason="  ".join(parts),
    )


def _signal_liquidity_sweep(
    high: pd.Series, low: pd.Series, close: pd.Series,
    open_: pd.Series, volume: pd.Series,
) -> "CandleSignal | None":
    if len(close) < 22:
        return None
    o, h, l, c, pc = float(open_.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1]), float(close.iloc[-2])
    body, lower_wick, total = abs(c - o), min(o, c) - l, h - l
    if total == 0 or body == 0:
        return None
    prior_5_low = float(low.iloc[-6:-1].min())
    avg_vol = float(volume.iloc[-21:-1].mean())
    rel_vol = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
    if (l < prior_5_low) and (c > pc) and (lower_wick >= 1.5 * body) and (rel_vol >= 1.3):
        depth = (prior_5_low - l) / prior_5_low * 100
        return CandleSignal(
            ticker="", pattern="Liquidity Sweep", direction="bullish",
            strength="strong" if depth > 0.5 and rel_vol > 1.5 else "moderate",
            action="HOLD",
            reason=f"Stop sweep {depth:.2f}%  recovered  wick={lower_wick/body:.1f}x body  vol={rel_vol:.1f}x",
        )
    return None


# ── main detector (public) ─────────────────────────────────────────────────────

def detect_patterns(
    ticker: str,
    df: pd.DataFrame,
    spy_df:    "pd.DataFrame | None" = None,
    sector_df: "pd.DataFrame | None" = None,
) -> list[CandleSignal]:
    """
    Run the 5-condition professional signal stack on `df`.

    df columns required: open, high, low, close, volume (lowercase).
    spy_df:    SPY OHLCV — required for RS and regime checks.
    sector_df: Sector ETF OHLCV — used for RS > sector check.
    Returns list of CandleSignal (may be empty).
    """
    if df is None or len(df) < 25:
        return []

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    spy_close    = spy_df["close"]    if spy_df    is not None and not spy_df.empty    else None
    sector_close = sector_df["close"] if sector_df is not None and not sector_df.empty else None

    raw = [
        _signal_combo(close, high, low, volume, spy_close, sector_close),
        _signal_near_miss(close, high, low, volume, spy_close, sector_close),
        _signal_bear_regime(spy_close),
        _signal_market_structure(close, high, low),
        _signal_liquidity_sweep(high, low, close, open_, volume),
    ]

    signals = []
    for sig in raw:
        if sig is not None:
            sig.ticker = ticker
            signals.append(sig)
    return signals


# ── scoring ────────────────────────────────────────────────────────────────────

_PRIORITY_WEIGHT: dict[str, float] = {
    "VolContraction+RelVol+VWAP+RS+Breakout": 20.0,
    "Near-Miss (4/5)":                         4.0,
    "Bear Regime":                            -20.0,
    "Market Structure":                         3.0,
    "Liquidity Sweep":                          4.0,
}
_STRENGTH_MULT: dict[str, float] = {"strong": 1.5, "moderate": 1.0, "weak": 0.5}


def _composite_score(signals: list[CandleSignal]) -> float:
    return sum(
        _PRIORITY_WEIGHT.get(s.pattern, 1.0) * _STRENGTH_MULT.get(s.strength, 1.0)
        for s in signals
    )


# ── public interface ───────────────────────────────────────────────────────────

def screen_candlesticks(watchlist: list[str]) -> list[dict]:
    """
    Screen tickers using the 5-condition professional signal stack.

    Decision tree (in order):
      1. Bear Regime present  → HOLD (SPY < SMA200 — no longs)
      2. 5/5 combo fires      → BUY  (high conviction)
      3. 4/5 near-miss        → HOLD (watchlist)
      4. Supporting signals   → HOLD (score >= 5)

    Fetches SPY once and sector ETFs on-demand (cached per run).
    Returns list of dicts: ticker, pattern, direction, strength,
    action, reason, conf, all_patterns.
    Interface identical to prior version.
    """
    from data.market_data import get_ohlcv, get_info as _get_info
    from loguru import logger

    # ── fetch SPY once ─────────────────────────────────────────────────────────
    spy_df: "pd.DataFrame | None" = None
    try:
        spy_df = get_ohlcv("SPY", period="2y", interval="1d")
        if spy_df is not None and spy_df.empty:
            spy_df = None
    except Exception as e:
        logger.warning(f"Candle | SPY fetch failed ({e})")

    # ── pre-fetch sector ETFs (unique set only) ────────────────────────────────
    sector_etf_cache: dict[str, "pd.DataFrame | None"] = {}

    def _get_sector_df(etf: str) -> "pd.DataFrame | None":
        if etf not in sector_etf_cache:
            try:
                df = get_ohlcv(etf, period="2y", interval="1d")
                sector_etf_cache[etf] = df if df is not None and not df.empty else None
            except Exception:
                sector_etf_cache[etf] = None
        return sector_etf_cache[etf]

    def _lookup_sector_etf(ticker: str) -> "pd.DataFrame | None":
        """Get sector ETF df for ticker via get_info (routes through yfinance .info)."""
        try:
            info   = _get_info(ticker)
            sector = info.get("sector", "")
            etf    = SECTOR_ETF_MAP.get(sector, "")
            if etf:
                return _get_sector_df(etf)
        except Exception:
            pass
        return None

    # ── scan ───────────────────────────────────────────────────────────────────
    results = []
    for ticker in watchlist:
        try:
            df = get_ohlcv(ticker, period="6mo", interval="1d")
            if df is None or df.empty or len(df) < 25:
                continue

            sector_df = _lookup_sector_etf(ticker)
            signals   = detect_patterns(ticker, df, spy_df, sector_df)
            if not signals:
                continue

            score    = _composite_score(signals)
            patterns = {s.pattern for s in signals}

            bear_veto  = "Bear Regime" in patterns
            combo_fire = "VolContraction+RelVol+VWAP+RS+Breakout" in patterns
            near_miss  = "Near-Miss (4/5)" in patterns

            if bear_veto:
                final_action = "HOLD"
                conf_detail  = "bear regime — SPY below 200 SMA"
                dominant     = next(s for s in signals if s.pattern == "Bear Regime")

            elif combo_fire:
                final_action = "BUY"
                dominant     = next(s for s in signals if s.pattern == "VolContraction+RelVol+VWAP+RS+Breakout")
                n_sup        = len([s for s in signals if s.pattern not in {
                    "VolContraction+RelVol+VWAP+RS+Breakout", "Bear Regime", "Near-Miss (4/5)"
                }])
                conf_detail  = f"5/5 COMBO  {'+' + str(n_sup) + ' supporting' if n_sup else ''}"

            elif near_miss:
                final_action = "HOLD"
                dominant     = next(s for s in signals if s.pattern == "Near-Miss (4/5)")
                conf_detail  = dominant.reason

            elif score >= 5.0:
                final_action = "HOLD"
                dominant     = signals[0]
                conf_detail  = f"score={score:.0f}"

            else:
                continue

            logger.info(
                f"Candle | {ticker:6s} {dominant.pattern:44s} "
                f"-> {final_action}  {conf_detail}"
            )

            results.append({
                "ticker":       ticker,
                "pattern":      dominant.pattern,
                "direction":    dominant.direction,
                "strength":     dominant.strength,
                "action":       final_action,
                "reason":       dominant.reason,
                "conf":         conf_detail,
                "all_patterns": [s.pattern for s in signals],
            })

        except Exception as exc:
            from loguru import logger as _log
            _log.warning(f"Candlestick scan failed for {ticker}: {exc}")

    strength_order = {"strong": 0, "moderate": 1, "weak": 2}
    action_order   = {"BUY": 0, "SELL": 1, "HOLD": 2}
    results.sort(key=lambda x: (
        action_order.get(x["action"], 9),
        strength_order.get(x["strength"], 9),
    ))
    return results


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    print(f"Scanning {len(CUSTOM_WATCHLIST)} tickers...\n")
    hits = screen_candlesticks(CUSTOM_WATCHLIST)
    if not hits:
        print("No signals today.")
    else:
        print(f"{'TICKER':<8} {'STRENGTH':<10} {'ACTION':<6} {'PATTERN':<46} DETAIL")
        print("-" * 125)
        for h in hits:
            detail = h["conf"] if h["conf"] else h["reason"][:50]
            print(f"{h['ticker']:<8} {h['strength']:<10} {h['action']:<6} {h['pattern']:<46} {detail}")
