"""
agents/signal_router.py — Deduplicates and ranks signals from all active strategies.

Phase 1 engines:
    momentum            (agents/momentum_screener.py)
    institutional_breakout (agents/institutional_breakout.py)

Conflict resolution rule (A — IB wins):
    When the same ticker fires from both momentum AND institutional_breakout,
    institutional_breakout takes priority.  One position is opened.  The
    journal entry is tagged 'institutional_breakout'.
    Rationale: both signals are usually describing the same phenomenon (strong
    trend + breakout) but IB has tighter entry conditions, cleaner stops, and
    a purpose-built exit profile.

Signal output format:
    {
        "ticker":   "PLTR",
        "strategy": "institutional_breakout",
        "score":    92.0,
        "entry":    145.50,
        "stop":     138.20,
        "atr":      3.65,
    }

The router also pre-computes RS ranks once per scan cycle so both engines
share the same universe rank table rather than each computing it independently.

Usage:
    from agents.signal_router import route_signals
    signals = route_signals(watchlist, regime=regime)
    # -> sorted list of signal dicts, deduped, IB priority
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# RS rank pre-computation (shared across engines per scan)
# ---------------------------------------------------------------------------

def _get_rs_ranks() -> dict[str, float]:
    """
    Fetch RS ranks for the full FMP universe.
    Returns {} gracefully when FMP is unavailable so downstream gates
    treat all stocks as unranked (rs_rank = -1) rather than crashing.
    """
    try:
        from data.fmp_data import compute_rs_ranks, get_rs_universe
        universe = get_rs_universe()
        if not universe:
            logger.warning("signal_router | RS universe empty — FMP may be misconfigured")
            return {}
        return compute_rs_ranks(universe)
    except Exception as exc:
        logger.warning("signal_router | RS ranks unavailable: {}", exc)
        return {}


# ---------------------------------------------------------------------------
# Convert strategy-native signal objects to a common dict format
# ---------------------------------------------------------------------------

def _from_momentum(r: Any) -> dict:
    """Convert a momentum screener result dict to the router's common format."""
    return {
        "ticker":   r["ticker"],
        "strategy": "momentum",
        "score":    float(r.get("weighted_score", 0)),
        "entry":    float(r.get("close", 0)),
        "stop":     None,   # momentum uses exit_manager's regime-based ATR stop
        "atr":      None,
    }


def _from_ib(sig: Any) -> dict:
    """Convert an IBSignal to the router's common format."""
    return {
        "ticker":   sig.ticker,
        "strategy": "institutional_breakout",
        "score":    sig.score,
        "entry":    sig.entry,
        "stop":     sig.stop_atr,   # 2 ATR initial stop
        "atr":      sig.atr,
    }


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def route_signals(
    watchlist: list[str],
    regime=None,
    run_momentum: bool = True,
    run_ib:       bool = True,
) -> list[dict]:
    """
    Run all active strategy engines, deduplicate results, and return a
    prioritised signal list.

    Args:
        watchlist    : tickers to scan
        regime       : RegimeState from market_regime.compute_regime()
        run_momentum : include momentum screener results
        run_ib       : include institutional breakout results

    Returns:
        List of signal dicts, sorted by score descending, deduplicated.
        Each dict has: ticker, strategy, score, entry, stop, atr.
    """
    start = time.time()

    # Pre-compute RS ranks once — shared by IB engine
    rs_ranks: dict[str, float] = {}
    if run_ib:
        logger.info("signal_router | computing RS ranks for IB engine...")
        rs_ranks = _get_rs_ranks()
        logger.info("signal_router | RS ranks: {} tickers", len(rs_ranks))

    raw_signals: dict[str, dict] = {}   # ticker -> best signal dict

    # ── Momentum screener ──────────────────────────────────────────────────
    if run_momentum:
        try:
            from agents.momentum_screener import screen as mom_screen
            mom_results = mom_screen(watchlist, regime=regime)
            for r in (mom_results or []):
                ticker = r.get("ticker", "").upper()
                if ticker:
                    raw_signals[ticker] = _from_momentum(r)
            logger.info("signal_router | momentum: {} signals", len(mom_results or []))
        except Exception as exc:
            logger.warning("signal_router | momentum screener failed: {}", exc)

    # ── Institutional Breakout ─────────────────────────────────────────────
    if run_ib:
        try:
            from agents.institutional_breakout import screen as ib_screen
            ib_results = ib_screen(watchlist, rs_ranks=rs_ranks, regime=regime)
            for sig in (ib_results or []):
                ticker = sig.ticker.upper()
                # IB WINS: overwrite momentum signal on conflict (rule A)
                if ticker in raw_signals:
                    logger.info(
                        "signal_router | {}: IB overrides momentum signal "
                        "(both fired — same phenomenon, IB has priority)",
                        ticker,
                    )
                raw_signals[ticker] = _from_ib(sig)
            logger.info("signal_router | institutional_breakout: {} signals", len(ib_results or []))
        except Exception as exc:
            logger.warning("signal_router | institutional_breakout failed: {}", exc)

    # ── Sort by score, log summary ─────────────────────────────────────────
    final = sorted(raw_signals.values(), key=lambda s: -s["score"])
    elapsed = round(time.time() - start, 1)

    if final:
        logger.info(
            "signal_router | {} total signals in {}s  "
            "IB={} momentum={}  top: {}",
            len(final), elapsed,
            sum(1 for s in final if s["strategy"] == "institutional_breakout"),
            sum(1 for s in final if s["strategy"] == "momentum"),
            [(s["ticker"], s["strategy"][:2], s["score"]) for s in final[:5]],
        )
    else:
        logger.info("signal_router | no signals this scan ({}s)", elapsed)

    return final
