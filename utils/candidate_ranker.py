"""
utils/candidate_ranker.py -- Relative-strength ranking for screened BUY candidates.

Instead of trading every signal that clears the score threshold, rank all
qualifying candidates against each other and only execute the top N.

Ranking dimensions (all percentile-ranked within the current candidate set
so the comparison is always apples-to-apples regardless of absolute values):

  1. Relative Volume      -- institutional participation; high = conviction
  2. RS vs SPY            -- stock outperforming the broad market
  3. Momentum Percentile  -- RSI rank (strategy-aware: higher=better in trends,
                             lower=better in mean reversion)
  4. ATR Profile          -- trend: sweet-spot expansion (40-60th pct);
                             mean_reversion: maximum compression
  5. Total Score          -- composite factor quality from screener

Sector strength and institutional (options) flow are supported as optional
bonus signals when the data is available in the result dict.

Tune MAX_TRADES_PER_SCAN to control how concentrated or diversified you want
each scan cycle to be.
"""
from __future__ import annotations

import math
from loguru import logger

# Maximum new positions opened per single scan run.
# Keep small: 1-2 forces genuine selectivity; 3 allows some diversification.
MAX_TRADES_PER_SCAN = 2

# Minimum score required to even enter the ranking pool
MIN_SCORE_FOR_RANKING = 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_rank(value: float, all_values: list[float]) -> float:
    """Percentile rank of value within all_values, returns 0.0–1.0.
    1.0 = highest in the set, 0.0 = lowest.
    Handles ties and single-element sets gracefully.
    """
    if len(all_values) <= 1:
        return 1.0
    below = sum(1 for v in all_values if v < value)
    equal = sum(1 for v in all_values if v == value)
    # Mid-rank for ties
    return (below + equal / 2) / len(all_values)


def _compute_rank_score(candidate: dict, strategy: str, pool: list[dict]) -> float:
    """
    Returns a composite rank score in [0, 1] for a single candidate relative
    to all other candidates in the pool.

    Weights differ by strategy:

    TREND FOLLOWING
      High relative volume   → big weight (institutional buying = fuel for move)
      Outperforming SPY      → big weight (sector / stock leadership)
      Higher RSI             → mild weight (momentum confirmation)
      ATR sweet spot         → mild weight (some expansion, not extended)
      Total factor score     → anchor weight

    MEAN REVERSION
      Low ATR rank           → big weight (maximum compression = coiled spring)
      Low RSI (most oversold)→ big weight (deepest oversold = best bounce energy)
      High total score       → big weight (quality matters when buying weakness)
      High rel volume        → mild weight (exhaustion volume is bullish)
      RS vs SPY              → small weight (less important for short-term bounces)
    """
    # Raw values
    rel_vol  = candidate.get("_rel_volume",   1.0)
    rs_spy   = candidate.get("_rs_vs_spy",    0.0)
    atr_rank = candidate.get("_atr_pct_rank", 50.0)
    rsi      = candidate.get("_rsi_14",       50.0)
    total    = candidate.get("total_score",    0.0)
    # Optional bonus signals
    opt_flow = candidate.get("_options_flow_score", 0.0)  # 0 if not available

    # Pool arrays for percentile ranking
    all_vol   = [c.get("_rel_volume",   1.0)  for c in pool]
    all_rs    = [c.get("_rs_vs_spy",    0.0)  for c in pool]
    all_atr   = [c.get("_atr_pct_rank", 50.0) for c in pool]
    all_rsi   = [c.get("_rsi_14",       50.0) for c in pool]
    all_total = [c.get("total_score",   0.0)  for c in pool]

    vol_pct   = _pct_rank(rel_vol, all_vol)
    rs_pct    = _pct_rank(rs_spy, all_rs)
    total_pct = _pct_rank(total, all_total)

    if strategy == "trend_following":
        # RSI: higher = more momentum (better in trends)
        rsi_pct = _pct_rank(rsi, all_rsi)

        # ATR sweet spot for trend breakouts: 35-60th percentile.
        # Not too compressed (no energy yet) and not over-expanded (chasing).
        # Score peaks at atr_rank=47, falls off toward 0 and 100.
        atr_score = 1.0 - abs(atr_rank - 47) / 53

        score = (
            0.30 * vol_pct    # relative volume  (most predictive of follow-through)
          + 0.25 * rs_pct     # RS vs SPY        (sector & stock leadership)
          + 0.20 * total_pct  # factor quality   (overall conviction)
          + 0.15 * rsi_pct    # momentum         (confirm price rising)
          + 0.10 * atr_score  # ATR profile      (breakout energy present but not chasing)
        )

    else:  # mean_reversion
        # ATR: lower rank = more compressed = better for reversion
        atr_compress = 1.0 - (atr_rank / 100)

        # RSI: LOWER = more oversold = better (flip the percentile)
        rsi_oversold = 1.0 - _pct_rank(rsi, all_rsi)

        score = (
            0.30 * atr_compress  # ATR compression  (coiled spring energy)
          + 0.25 * rsi_oversold  # RSI oversold      (deepest oversold = best bounce)
          + 0.25 * total_pct     # factor quality    (fundamentals weighted 2× in MR prompt)
          + 0.15 * vol_pct       # volume surge      (exhaustion volume at lows = bullish)
          + 0.05 * rs_pct        # RS vs SPY         (minor; stock will catch up on bounce)
        )

    # Optional: options flow bonus (±5% adjustment when available)
    if opt_flow != 0.0:
        # Normalise to ±0.05 additive bonus/penalty
        score += 0.05 * math.tanh(opt_flow)

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_and_filter(
    results: list[dict],
    top_n: int = MAX_TRADES_PER_SCAN,
    min_score: int = MIN_SCORE_FOR_RANKING,
) -> list[dict]:
    """
    Filter screened results to BUY candidates, rank them by composite
    relative-strength score, and return the top N.

    Args:
        results   : output of stock_screener.screen()
        top_n     : maximum candidates to return (default MAX_TRADES_PER_SCAN)
        min_score : minimum screener score to enter the ranking pool

    Returns:
        List of up to top_n result dicts, each augmented with:
          rank_score  -- composite RS score (0-1)
          rank_pos    -- position in the ranking (1 = best)
    """
    # Build candidate pool: any actionable signal above minimum score threshold.
    # stock_screener emits "BUY"/"HOLD"/"SKIP"; unified_screener emits
    # "STRONG BUY"/"CONFIRMED"/"WATCH"/"SKIP" — accept all buy-side variants.
    BUY_SIGNALS = {"BUY", "STRONG BUY", "CONFIRMED", "STRONG_BUY"}
    pool = [
        r for r in results
        if r.get("signal", "").upper().replace(" ", "_") in
           {s.upper().replace(" ", "_") for s in BUY_SIGNALS}
        and r.get("score", 0) >= min_score
    ]

    if not pool:
        return []

    if len(pool) == 1:
        pool[0]["rank_score"] = 1.0
        pool[0]["rank_pos"]   = 1
        logger.info(f"Ranker | 1 candidate, no ranking needed: {pool[0]['ticker']}")
        return pool

    strategy = pool[0].get("strategy", "trend_following")

    # Score every candidate relative to the full pool
    for candidate in pool:
        candidate["rank_score"] = _compute_rank_score(candidate, strategy, pool)

    ranked = sorted(pool, key=lambda x: x["rank_score"], reverse=True)

    # Log the full ranking table before filtering
    logger.info(
        f"Ranker | {len(ranked)} BUY candidates [{strategy}]  "
        f"→ selecting top {min(top_n, len(ranked))}"
    )
    for i, c in enumerate(ranked, 1):
        c["rank_pos"] = i
        selected = "✓" if i <= top_n else "✗"
        logger.info(
            f"  {selected} #{i} {c['ticker']:6s}  "
            f"rank={c['rank_score']:.3f}  "
            f"score={c['score']}  total={c['total_score']:+.2f}  "
            f"relVol={c.get('_rel_volume', 0):.1f}x  "
            f"RS={c.get('_rs_vs_spy', 0):+.3f}  "
            f"RSI={c.get('_rsi_14', 0):.0f}  "
            f"ATR_rank={c.get('_atr_pct_rank', 0):.0f}"
        )

    return ranked[:top_n]
