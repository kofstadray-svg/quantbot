"""
agents/stock_screener.py -- Regime-adaptive orthogonal-factor stock screener.

The screener selects a different entry strategy based on the current market
regime, rather than applying fixed scoring rules regardless of context.

Strategy selection (from market_regime.market_type + risk_regime + vol_regime):
  ┌────────────────────┬────────────────────────────────────────────────────┐
  │ Strategy           │ When active                                        │
  ├────────────────────┼────────────────────────────────────────────────────┤
  │ trend_following    │ market_type=trending, risk_on, vol not extreme     │
  │ mean_reversion     │ market_type=mean_reverting or choppy, risk_on,     │
  │                    │ vol not high/extreme                               │
  │ disabled           │ bear/strong_bear, extreme vol, or risk_off in      │
  │                    │ choppy/mean_reverting market                       │
  └────────────────────┴────────────────────────────────────────────────────┘

Strategy differences:
  TREND FOLLOWING   -- rewards RSI 50-65 (rising momentum), ADX>25, golden
                       cross, positive MA50 slope, stock outperforming SPY.
                       Penalises RSI>70 (extended/chasing) and ATR expansion.

  MEAN REVERSION    -- rewards RSI<40 (oversold bounce candidate), ATR
                       compression (coiled), volume surge at lows (exhaustion).
                       trend_score weight halved (stock may be in short-term
                       downtrend -- that's the OPPORTUNITY, not a disqualifier).
                       fundamental_score weight doubled (need quality for bounce).

Feature set designed for minimal inter-factor correlation:
  TREND       50MA slope + ADX(14) + golden cross
  MOMENTUM    RSI(14) only
  VOLATILITY  ATR percentile rank (1-year)
  LIQUIDITY   Relative volume (20d) + OBV slope + RS vs SPY
  FUNDAMENTAL Revenue growth + trailing PE
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import ta
from loguru import logger
from data.market_data import get_ohlcv, get_info
from utils.claude_client import ask_claude_json
from agents.prompt_builder import build_prompt, get_system_prompt  # noqa: F401  (re-export)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def select_strategy(regime) -> tuple[str, str | None]:
    """
    Map market regime → (strategy_name, prompt_or_None).
    Returns ("disabled", None) when entries should be suppressed entirely.

    Selection logic:
      extreme vol           → disabled (unpredictable, slippage destroys edge)
      bear / strong_bear    → disabled (trend entries fail, reversion too risky)
      trending + risk_on    → trend_following
      trending + risk_off   → trend_following (stricter threshold in regime_params)
      choppy/mean_reverting + risk_on + vol not high/extreme → mean_reversion
      choppy/mean_reverting + risk_off                       → disabled
      choppy/mean_reverting + high vol                       → disabled
    """
    if regime.vol_regime == "extreme":
        return "disabled", None
    if regime.trend_regime in ("bear", "strong_bear"):
        return "disabled", None
    if regime.market_type == "trending":
        return "trend_following", build_prompt("trend_following")
    # choppy or mean_reverting
    if regime.risk_regime == "risk_off" or regime.vol_regime == "high":
        return "disabled", None
    return "mean_reversion", build_prompt("mean_reversion")


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _obv_slope(close: pd.Series, volume: pd.Series, lookback: int = 14) -> str:
    """OBV trend over last `lookback` bars: 'rising' | 'flat' | 'falling'."""
    obv_val = 0.0
    obv = []
    for i in range(len(close)):
        if i == 0:
            obv.append(0.0)
            continue
        if close.iloc[i] > close.iloc[i - 1]:
            obv_val += float(volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_val -= float(volume.iloc[i])
        obv.append(obv_val)
    recent = pd.Series(obv).iloc[-lookback:]
    slope = float(np.polyfit(range(len(recent)), recent.values, 1)[0])
    if slope > obv_val * 0.001:
        return "rising"
    if slope < -obv_val * 0.001:
        return "falling"
    return "flat"


def _atr_percentile(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> float:
    """
    ATR percentile rank: where does today's ATR(14) sit in its 1-year range?
    Returns 0-100 (0 = lowest ATR in a year, 100 = highest).
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    if len(atr) < 2:
        return 50.0
    current = float(atr.iloc[-1])
    pct_rank = float((atr < current).sum() / len(atr) * 100)
    return round(pct_rank, 1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> float:
    """ADX(14) -- trend strength, direction-agnostic."""
    try:
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close,
                                         window=period)
        v = float(adx_ind.adx().iloc[-1])
        return round(v, 2) if not np.isnan(v) else 20.0
    except Exception:
        return 20.0


def _rs_vs_spy(ticker_close: pd.Series) -> float:
    """
    20-day annualised slope of the stock/SPY price ratio.
    Positive = outperforming, negative = lagging.
    Returns % per day (e.g. +0.3 means stock gains 0.3% per day vs SPY).
    """
    try:
        spy_df = get_ohlcv("SPY", period="3mo", interval="1d")
        if spy_df is None or spy_df.empty:
            return 0.0
        spy = spy_df["close"].squeeze()
        # Align on common dates
        ratio = ticker_close / spy.reindex(ticker_close.index).ffill()
        ratio = ratio.dropna().iloc[-21:]
        if len(ratio) < 5:
            return 0.0
        slope = float(np.polyfit(range(len(ratio)), ratio.values, 1)[0])
        # Normalise to % per day relative to ratio level
        slope_pct = slope / float(ratio.iloc[0]) * 100
        return round(slope_pct, 4)
    except Exception:
        return 0.0


def _ma50_slope(close: pd.Series) -> float:
    """Slope of 50MA over last 10 bars, expressed as % of current price per bar."""
    ma50 = close.rolling(50).mean().dropna()
    if len(ma50) < 10:
        return 0.0
    recent = ma50.iloc[-10:]
    slope = float(np.polyfit(range(10), recent.values, 1)[0])
    return round(slope / float(close.iloc[-1]) * 100, 4)


def _compute_indicators(ticker: str) -> dict:
    df = get_ohlcv(ticker, period="1y", interval="1d")
    if df is None or df.empty or len(df) < 50:
        df = get_ohlcv(ticker, period="6mo", interval="1d")
    if df is None or df.empty or len(df) < 20:
        return {}

    close  = df["close"].squeeze()
    volume = df["volume"].squeeze()
    high   = df["high"].squeeze()
    low    = df["low"].squeeze()

    # -- Trend --
    ma50      = close.rolling(50).mean()
    ma200     = close.rolling(200).mean() if len(close) >= 200 else None
    golden    = bool(ma200 is not None and
                     not np.isnan(float(ma200.iloc[-1])) and
                     float(ma50.iloc[-1]) > float(ma200.iloc[-1]))
    ma50_slp  = _ma50_slope(close)
    adx_val   = _adx(high, low, close)

    # -- Momentum (RSI only) --
    rsi = round(float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]), 2)

    # -- Volatility (ATR percentile) --
    atr_rank = _atr_percentile(high, low, close)

    # -- Liquidity --
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    rel_vol    = round(float(volume.iloc[-1] / avg_vol_20), 2) if avg_vol_20 else 1.0
    obv        = _obv_slope(close, volume, lookback=14)
    rs_slope   = _rs_vs_spy(close)

    # -- Fundamental --
    # OpenBB (cached 24h) supplies normalised PE / forward PE / P/B / beta /
    # market cap via yfinance under the hood; we keep info.get("revenueGrowth")
    # since OpenBB's free yfinance profile doesn't surface it. On OpenBB outage
    # `fund` falls back to {} and we still get revenue_growth + a usable pe_ratio
    # from the original info dict.
    info = get_info(ticker)
    fund = {}  # OpenBB removed — fundamentals not available without openbb_data or {}

    return {
        "ticker":           ticker,
        "price":            round(float(close.iloc[-1]), 2),
        # Trend
        "golden_cross":     golden,
        "ma50_slope_pct":   ma50_slp,
        "adx_14":           adx_val,
        # Momentum
        "rsi_14":           rsi,
        # Volatility
        "atr_pct_rank":     atr_rank,
        # Liquidity
        "rel_volume":       rel_vol,
        "obv_slope":        obv,
        "rs_vs_spy_slope":  rs_slope,
        # Fundamental (OpenBB-augmented; falls back to yfinance.info on outage)
        "pe_ratio":         fund.get("pe_ratio") if fund.get("pe_ratio") is not None
                            else info.get("trailingPE"),
        "forward_pe":       fund.get("forward_pe"),
        "price_to_book":    fund.get("price_to_book"),
        "beta":             fund.get("beta"),
        "market_cap":       fund.get("market_cap"),
        "revenue_growth":   info.get("revenueGrowth"),
    }


# ---------------------------------------------------------------------------
# Signal derivation
# ---------------------------------------------------------------------------

def _derive_signal(weights: dict, buy_threshold: float = 3.5) -> tuple[str, float, int]:
    """Convert factor weights to (signal, total_score, scaled_score).

    Factor key renamed: volume_score -> liquidity_score for clarity.
    Both old and new key names accepted for backward compat.
    """
    total = (
        weights.get("trend_score",       0.0) +
        weights.get("momentum_score",    0.0) +
        weights.get("volatility_score",  0.0) +
        weights.get("liquidity_score",   weights.get("volume_score", 0.0)) +
        weights.get("fundamental_score", 0.0)
    )
    hold_threshold = buy_threshold * 0.4
    if total >= buy_threshold:
        signal = "BUY"
    elif total >= hold_threshold:
        signal = "HOLD"
    else:
        signal = "SKIP"
    # Map [-8, +8] -> [1, 10]
    scaled = max(1, min(10, round((total + 8) * 9 / 16 + 1)))
    return signal, total, scaled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen(watchlist: list[str], regime=None) -> list[dict]:
    """Screen a list of tickers with regime-adaptive factor scoring.

    Strategy is selected from the market regime before any ticker is scored:
      - trend_following  : trending markets, risk-on
      - mean_reversion   : choppy/mean-reverting markets, risk-on, normal vol
      - disabled         : bear markets, extreme vol, or risk-off + choppy

    Returns a ranked list of result dicts (empty if strategy is disabled).
    """
    if regime is None:
        from agents.market_regime import compute_regime
        regime = compute_regime()

    buy_threshold = regime.buy_threshold
    size_mult     = regime.size_mult
    regime_label  = regime.label

    # ── Strategy selection ──────────────────────────────────────────────────
    strategy_name, strategy_prompt = select_strategy(regime)

    if strategy_name == "disabled":
        logger.info(
            f"Screener DISABLED for regime [{regime_label}]  "
            f"(market_type={regime.market_type}  vol={regime.vol_regime}  "
            f"risk={regime.risk_regime}) -- no entries until conditions improve."
        )
        return []

    logger.info(
        f"Screener strategy: {strategy_name.upper()}  "
        f"[{regime_label}  thresh={buy_threshold}  size={size_mult}x]"
    )

    # Allow autoresearch prompt to override (backward-compat), otherwise use
    # the regime-selected strategy prompt.
    try:
        from utils.autoresearch import load_active_prompt
        base_prompt = load_active_prompt()
    except Exception:
        base_prompt = strategy_prompt

    results = []
    for ticker in watchlist:
        logger.info(f"Screening {ticker}...")
        try:
            indicators = _compute_indicators(ticker)
            if not indicators:
                continue

            raw     = ask_claude_json(base_prompt, json.dumps(indicators))
            weights = json.loads(raw)
            signal, total, score = _derive_signal(weights, buy_threshold)

            result = {
                "ticker":            ticker,
                "price":             indicators["price"],
                "signal":            signal,
                "score":             score,
                "total_score":       round(total, 3),
                "trend_score":       weights.get("trend_score",       0.0),
                "momentum_score":    weights.get("momentum_score",    0.0),
                "volatility_score":  weights.get("volatility_score",  0.0),
                "liquidity_score":   weights.get("liquidity_score",
                                                  weights.get("volume_score", 0.0)),
                "volume_score":      weights.get("liquidity_score",
                                                  weights.get("volume_score", 0.0)),
                "fundamental_score": weights.get("fundamental_score", 0.0),
                "confidence":        weights.get("confidence",        0.5),
                "buy_threshold":     buy_threshold,
                "size_mult":         size_mult,
                "regime_label":      regime_label,
                "strategy":          strategy_name,
                # ── Raw indicators for candidate ranking (prefixed _) ──────
                "_rel_volume":       indicators.get("rel_volume",      1.0),
                "_rs_vs_spy":        indicators.get("rs_vs_spy_slope", 0.0),
                "_atr_pct_rank":     indicators.get("atr_pct_rank",   50.0),
                "_rsi_14":           indicators.get("rsi_14",         50.0),
                "_adx_14":           indicators.get("adx_14",         20.0),
            }
            results.append(result)
            logger.info(
                f"{ticker:6s} total={total:+.2f} "
                f"(T={result['trend_score']:+.1f} "
                f"M={result['momentum_score']:+.1f} "
                f"Vt={result['volatility_score']:+.1f} "
                f"L={result['liquidity_score']:+.1f} "
                f"F={result['fundamental_score']:+.1f}) "
                f"conf={result['confidence']:.2f} "
                f"thresh={buy_threshold} [{strategy_name}] -> {signal}"
            )
        except Exception as exc:
            logger.error(f"{ticker} screening failed: {exc}")

    results.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
    return results


if __name__ == "__main__":
    watchlist = ["AAPL", "NVDA", "MSFT", "META", "TSLA"]
    ranked = screen(watchlist)
    for r in ranked:
        print(
            f"{r['ticker']:6s}  total={r['total_score']:+.2f}  "
            f"score={r['score']}  {r['signal']:4s}  conf={r['confidence']:.2f}"
        )
