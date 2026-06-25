"""
data/exit_state.py — Helpers to read/write per-ticker state in exit_state.json.

Thin wrapper around the same JSON file that exit_manager.py uses, so the
institutional_breakout scanner can tag a new position with its strategy +
ATR values at trade time, enabling exit_manager to route it to the correct
exit profile (ATR-based vs percentage-based).

All functions are fail-soft — any exception is logged and silently swallowed
so a state-tagging failure never blocks order execution.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from loguru import logger

_STATE_PATH = Path(__file__).parent.parent / "data" / "exit_state.json"


def _load() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def tag_strategy(
    ticker: str,
    *,
    strategy: str,
    entry_price: float,
    atr: float,
) -> None:
    """
    Write (or update) the strategy tag and ATR for `ticker` in exit_state.json.
    Called immediately after a successful order placement by the IB scanner.

    exit_manager.py reads state[ticker]["strategy"] to route to the correct
    exit profile.  If the ticker already has a state entry (from a prior
    position cycle), its strategy/atr fields are updated without disturbing
    the partial_taken flags or trail_high.
    """
    try:
        state = _load()
        if ticker not in state:
            state[ticker] = {
                "entry_date":    date.today().isoformat(),
                "entry_price":   entry_price,
                "partial_taken":  False,
                "partial2_taken": False,
                "ib_partial1_taken": False,
                "ib_partial2_taken": False,
                "trail_high":    entry_price,
                "rsi_peak":      0.0,
            }
        state[ticker]["strategy"]    = strategy
        state[ticker]["entry_atr"]   = atr
        state[ticker]["entry_price"] = entry_price   # ensure correct value
        _save(state)
        logger.info(
            "exit_state | {} tagged: strategy={} entry={:.2f} ATR={:.2f}",
            ticker, strategy, entry_price, atr,
        )
    except Exception as exc:
        logger.warning("exit_state | tag_strategy({}) failed: {}", ticker, exc)


def get_strategy(ticker: str) -> str:
    """Return the strategy tag for `ticker`, defaulting to 'momentum'."""
    try:
        state = _load()
        return state.get(ticker, {}).get("strategy", "momentum")
    except Exception:
        return "momentum"
