"""
utils/risk.py — Hard stop-loss and position-sizing guards.
Every agent must call check_position() before placing any order.
"""
from __future__ import annotations
import os
from loguru import logger
from config import MAX_POSITION_SIZE_USD, DAILY_LOSS_LIMIT_USD

_daily_loss: float = 0.0
_halted: bool = False

# Portfolio drawdown halt — compares live equity to equity at last daily reset.
# Resets to None every morning so one bad day cannot permanently brick the bot.
_session_start_equity: float | None = None
DRAWDOWN_LIMIT_PCT: float = float(os.getenv("DRAWDOWN_LIMIT_PCT", "0.15"))  # 15% default


def halt_trading(reason: str) -> None:
    """Latch a global trading halt (cleared at next daily reset)."""
    global _halted
    if not _halted:
        logger.critical(f"TRADING HALTED: {reason}")
    _halted = True


def is_halted() -> bool:
    return _halted


def reset_daily_loss() -> None:
    """Call this at the start of each trading day."""
    global _daily_loss, _session_start_equity, _halted
    _daily_loss = 0.0
    _session_start_equity = None   # will be set on first drawdown check
    _halted = False
    logger.info("Daily loss counter reset to $0.")


def check_portfolio_drawdown(current_equity: float) -> str | None:
    """
    Compare current equity against the session-start equity.
    Returns an error string if drawdown limit is breached, else None.
    Call once per loop with the live Alpaca equity value.
    """
    global _session_start_equity
    if _session_start_equity is None:
        _session_start_equity = current_equity
        logger.info(f"Session start equity recorded: ${current_equity:,.2f}")
        return None
    if _session_start_equity <= 0:
        return None
    drawdown = (_session_start_equity - current_equity) / _session_start_equity
    if drawdown >= DRAWDOWN_LIMIT_PCT:
        msg = (
            f"DRAWDOWN HALT: {drawdown:.1%} loss "
            f"(${_session_start_equity:,.2f} → ${current_equity:,.2f}) "
            f"exceeds {DRAWDOWN_LIMIT_PCT:.0%} limit"
        )
        logger.critical(msg)
        return msg
    return None


def record_loss(amount: float) -> None:
    """Record a realized loss (pass positive USD value)."""
    global _daily_loss
    _daily_loss += amount
    logger.warning(f"Loss recorded: ${amount:.2f}  |  Daily total: ${_daily_loss:.2f}")


def daily_limit_hit() -> bool:
    return _daily_loss >= DAILY_LOSS_LIMIT_USD


def check_position(notional_usd: float) -> bool:
    """
    Returns True if the trade is within risk limits and safe to place.
    Returns False and logs the reason if it should be blocked.
    Single chokepoint — covers both market_buy and smart_buy paths.
    """
    if _halted:
        logger.error("TRADE BLOCKED — trading halted (drawdown/daily-loss breach).")
        return False
    if daily_limit_hit():
        logger.error(
            f"TRADE BLOCKED — daily loss limit ${DAILY_LOSS_LIMIT_USD} reached "
            f"(current: ${_daily_loss:.2f})."
        )
        return False
    if notional_usd > MAX_POSITION_SIZE_USD:
        logger.error(
            f"TRADE BLOCKED — position ${notional_usd:.2f} exceeds max "
            f"${MAX_POSITION_SIZE_USD}."
        )
        return False
    return True


def kelly_fraction(win_prob: float, win_pct: float, loss_pct: float = 1.0) -> float:
    """
    Full Kelly fraction. Use half-Kelly in practice.
    win_prob : probability of winning (0-1)
    win_pct  : fraction gained on a win  (e.g. 0.30 for 30%)
    loss_pct : fraction lost on a loss   (e.g. 1.0  for 100%)
    """
    if loss_pct == 0 or win_pct == 0:
        return 0.0
    q = 1 - win_prob
    f = (win_prob / loss_pct) - (q / win_pct)
    return max(0.0, f / 2)  # half-Kelly
