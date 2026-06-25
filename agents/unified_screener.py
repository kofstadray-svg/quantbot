"""
agents/unified_screener.py -- Two-layer confirmation screener.

Pipeline:
  STAGE 1 -- Momentum screener (fast, rule-based, no API)
    Weighted probabilistic scoring (0-100 pts) across 5 factor groups:
      1. Relative Strength  (35 pts): EMA9>21, RSI zone, near HOD
      2. Volume Expansion   (25 pts): rel-vol, abs vol, float rotation
      3. Breakout Confirm   (20 pts): gap size, entry timing setups
      4. VWAP Structure     (12 pts): price vs VWAP
      5. Volatility / Float  (8 pts): ATR%, small float
    Tiers: STRONG BUY >=85 | BUY >=70 | WATCH >=55 | SKIP <55

  STAGE 2 -- Claude AI screener (only runs on STRONG BUY + BUY from Stage 1)
    Output: score 1-10 across RSI, MACD, Golden Cross, Stoch, OBV, BBands

  CONFIDENCE BLEND:
    confidence = 0.60 x (mom_score/100) + 0.40 x (ai_score/10)
    Falls back to mom_score alone when AI is unavailable.

  UNIFIED SIGNALS:
    STRONG BUY  confidence >= 0.80 AND entry timing confirmed
    CONFIRMED   confidence >= 0.68  (both layers agree, no entry setup yet)
    WATCH       confidence >= 0.52  (one layer or borderline both)
    SKIP        confidence <  0.52

  SIZE MULT:
    Passed through from momentum layer (1.00 / 0.75 / 0.50 / 0.00).
    Further scaled by AI confidence when both layers are available.
"""
from __future__ import annotations
from loguru import logger

from agents.momentum_screener import screen as momentum_screen
from agents.stock_screener    import _compute_indicators, ask_claude_json, get_system_prompt
from agents.market_regime     import compute_regime, regime_adjusted_thresholds
import json


# -- Default confidence thresholds (overridden per-run by regime) ---------------

CONF_STRONG_BUY = 0.80
CONF_CONFIRMED  = 0.68
CONF_WATCH      = 0.52


# -- Stage 2: Claude AI analysis (single ticker) --------------------------------

def _claude_score(ticker: str, regime=None) -> dict | None:
    """Run Claude AI analysis on a single ticker. Returns score dict or None.

    regime is forwarded to get_system_prompt() so the correct prompt
    (trend-following vs mean-reversion) is selected for the current market.
    """
    try:
        indicators = _compute_indicators(ticker)
        if not indicators:
            return None
        prompt = get_system_prompt(regime)
        raw    = ask_claude_json(prompt, json.dumps(indicators))
        result = json.loads(raw)
        result["ticker"] = ticker
        result["price"]  = indicators["price"]
        return result
    except Exception as e:
        logger.warning(f"unified_screener | Claude analysis failed for {ticker}: {e}")
        return None


# -- Combined signal logic ------------------------------------------------------

def _combine(
    mom: dict,
    ai:  dict | None,
    conf_strong_buy: float = CONF_STRONG_BUY,
    conf_confirmed:  float = CONF_CONFIRMED,
    conf_watch:      float = CONF_WATCH,
    size_cap:        float = 1.00,
) -> tuple[str, float, float]:
    """
    Blend momentum score and AI score into a unified confidence value.

    Confidence thresholds and size_cap can be overridden by the calling
    screen() function when a regime is in effect.

    Returns
    -------
    signal      : str    STRONG BUY / CONFIRMED / WATCH / SKIP / BREAKDOWN
    confidence  : float  0.0 - 1.0
    size_mult   : float  0.00 / 0.50 / 0.75 / 1.00  (capped by size_cap)
    """
    if mom.get("breakdown"):
        return "BREAKDOWN", 0.0, 0.0

    mom_conf = mom.get("weighted_score", 0) / 100.0

    if ai is not None:
        ai_conf    = (ai.get("score", 0) or 0) / 10.0
        confidence = round(0.60 * mom_conf + 0.40 * ai_conf, 3)
    else:
        confidence = round(mom_conf, 3)

    entry = bool(mom.get("entry_timing"))

    if confidence >= conf_strong_buy and entry:
        return "STRONG BUY", confidence, round(min(1.00, size_cap), 3)
    if confidence >= conf_confirmed:
        return "CONFIRMED",  confidence, round(min(0.75, size_cap), 3)
    if confidence >= conf_watch:
        return "WATCH",      confidence, round(min(0.50, size_cap), 3)
    return "SKIP", confidence, 0.00


# -- Public screen() function ---------------------------------------------------

def screen(watchlist: list[str], claude_on_watch: bool = False) -> list[dict]:
    """
    Run the two-layer screener on a watchlist.

    Regime is computed automatically (30-minute cache) and used to:
      - Adjust momentum score thresholds (strong_bull lowers bar, bear raises it)
      - Adjust unified confidence thresholds
      - Cap position size_mult
      - Veto longs in strong_bear / bear+extreme-vol regimes

    Args:
        watchlist       : list of ticker symbols
        claude_on_watch : if True, also run Claude AI on WATCH-level signals
                          (slower but catches near-misses); default False

    Returns:
        list of result dicts, sorted by confidence (best first)
    """
    results = []

    # -- Compute regime once for this scan run ---------------------------------
    try:
        regime = compute_regime()
        rt     = regime_adjusted_thresholds(regime)
        logger.info(
            f"Unified screener | Regime: {rt['regime_label']}  "
            f"score_thresholds=SB:{rt['score_strong_buy']}/B:{rt['score_buy']}/W:{rt['score_watch']}  "
            f"conf_thresholds=SB:{rt['conf_strong_buy']}/C:{rt['conf_confirmed']}/W:{rt['conf_watch']}  "
            f"size_cap={rt['size_cap']:.2f}  veto_longs={rt['veto_longs']}  "
            f"breadth_mult={regime.breadth_mult:.2f}"
        )
    except Exception as e:
        logger.warning(f"Unified screener | Regime fetch failed ({e}) -- using static defaults.")
        regime = None
        rt     = dict(
            conf_strong_buy = CONF_STRONG_BUY,
            conf_confirmed  = CONF_CONFIRMED,
            conf_watch      = CONF_WATCH,
            size_cap        = 1.00,
            veto_longs      = False,
            regime_label    = "unknown (fallback)",
        )

    logger.info(f"Unified screener | Stage 1: momentum scan ({len(watchlist)} tickers)...")
    momentum_results = momentum_screen(watchlist, regime=regime)

    stage2_tickers = {
        r["ticker"]: r for r in momentum_results
        if r["signal"] in ("STRONG BUY", "BUY")
        or (claude_on_watch and r["signal"] == "WATCH")
    }

    skipped = len(momentum_results) - len(stage2_tickers)
    logger.info(
        f"Unified screener | Stage 1 complete: "
        f"{len(stage2_tickers)} advance to Claude AI, {skipped} skipped."
    )

    if not stage2_tickers:
        logger.info("Unified screener | No tickers passed Stage 1 -- done.")
        return []

    logger.info(f"Unified screener | Stage 2: Claude AI analysis ({len(stage2_tickers)} tickers)...")

    for ticker, mom in stage2_tickers.items():
        logger.info(f"Unified screener | Claude analyzing {ticker}...")
        ai = _claude_score(ticker, regime=regime)

        unified, confidence, size_mult = _combine(
            mom, ai,
            conf_strong_buy = rt["conf_strong_buy"],
            conf_confirmed  = rt["conf_confirmed"],
            conf_watch      = rt["conf_watch"],
            size_cap        = rt["size_cap"],
        )
        entry = mom.get("entry_timing", [])

        result = {
            # Core signal
            "ticker":          ticker,
            "unified_signal":  unified,
            "confidence":      confidence,
            "size_mult":       size_mult,
            # Momentum layer
            "mom_signal":      mom["signal"],
            "mom_score":       mom.get("weighted_score"),
            "mom_passed":      mom["passed"],
            "mom_total":       mom["total"],
            "mom_failures":    mom.get("failures", []),
            "score_breakdown": mom.get("score_breakdown", {}),
            "entry_timing":    entry,
            # Claude AI layer
            "ai_signal":       ai.get("signal")  if ai else None,
            "ai_score":        ai.get("score")   if ai else None,
            "ai_reason":       ai.get("reason")  if ai else "Claude analysis unavailable",
            # Price data
            "price":           mom.get("price"),
            "vwap":            mom.get("vwap"),
            "rsi":             mom.get("rsi"),
            "rel_volume":      mom.get("rel_volume"),
            "gap_pct":         mom.get("gap_pct"),
            "atr_pct":         mom.get("atr_pct"),
            # Regime context (for logging, downstream sizing, audit trail)
            "regime_label":    rt["regime_label"],
            "regime_size_cap": rt["size_cap"],
            "breadth_mult":    regime.breadth_mult if regime else 1.0,
            # ── Backward-compat aliases (candidate_ranker / main.py) ─────────
            # "signal"      → "BUY" for any actionable signal so rank_and_filter
            #                 sees it; keep original in unified_signal.
            # "score"       → integer 0-100 (mom weighted score) for min-score gate.
            # "total_score" → float used by rank_and_filter for cross-stock ranking.
            "signal":          ("BUY" if unified in ("STRONG BUY", "BUY") else unified),
            "score":           round(mom.get("weighted_score", 0)),
            "total_score":     float(mom.get("weighted_score", 0.0)),
        }

        entry_str = ", ".join(entry) if entry else "none"
        ai_str    = f"AI={ai.get('signal','?')}" + f"({ai.get('score','?')})" if ai else "AI=n/a"
        logger.info(
            f"Unified | {ticker:<6} [{unified:12s}] "
            f"conf={confidence:.2f}  x{size_mult:.2f}  "
            f"Mom={mom['signal']}({mom.get('weighted_score',0):.0f}/100)  {ai_str}  "
            f"Entry=[{entry_str}]  -- {result['ai_reason']}"
        )

        results.append(result)

    results.sort(key=lambda x: (x["confidence"], x["ai_score"] or 0), reverse=True)
    return results


# -- Convenience summary -------------------------------------------------------

def summary(results: list[dict]) -> None:
    """Print a clean summary table of unified screener results."""
    strong     = [r for r in results if r["unified_signal"] == "STRONG BUY"]
    confirmed  = [r for r in results if r["unified_signal"] == "CONFIRMED"]
    watches    = [r for r in results if r["unified_signal"] == "WATCH"]
    breakdowns = [r for r in results if r["unified_signal"] == "BREAKDOWN"]

    print("=" * 70)
    print("  UNIFIED SCREENER RESULTS  (ranked by confidence)")
    print("=" * 70)

    if strong:
        print(f"\n  STRONG BUY ({len(strong)}) -- call order fired  [x1.00 size]")
        for r in strong:
            print(f"    {r['ticker']:<6}  conf={r['confidence']:.2f}  "
                  f"mom={r['mom_score']:.0f}/100  AI={r['ai_score']}/10  "
                  f"Entry={','.join(r['entry_timing'])}  -- {r['ai_reason']}")

    if confirmed:
        print(f"\n  CONFIRMED ({len(confirmed)}) -- both layers agree, awaiting entry  [x0.75 size]")
        for r in confirmed:
            print(f"    {r['ticker']:<6}  conf={r['confidence']:.2f}  "
                  f"mom={r['mom_score']:.0f}/100  AI={r['ai_score']}/10  "
                  f"-- {r['ai_reason']}")

    if watches:
        print(f"\n  WATCH ({len(watches)}) -- borderline, monitor  [x0.50 size if traded]")
        for r in watches:
            print(f"    {r['ticker']:<6}  conf={r['confidence']:.2f}  "
                  f"Mom={r['mom_signal']}  AI={r['ai_signal']}({r['ai_score']})  "
                  f"-- {r['ai_reason']}")

    if breakdowns:
        print(f"\n  BREAKDOWN ({len(breakdowns)}) -- put order fired")
        for r in breakdowns:
            print(f"    {r['ticker']:<6}  RSI={r.get('rsi','?')}  "
                  f"Gap={r.get('gap_pct','?'):+}%  RelVol={r.get('rel_volume','?')}x")

    if not any([strong, confirmed, watches, breakdowns]):
        print("\n  No signals today.")

    print("=" * 70)
