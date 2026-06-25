"""
agents/options_flow.py — Day 31-60 of the book.
Watches unusual options activity (whale alerts) and decides whether to follow.
Uses the Unusual Whales public API (free tier available).
"""
from __future__ import annotations
import json
import httpx
from loguru import logger
from utils.claude_client import ask_claude_json
from brokers.alpaca import market_buy
from utils.risk import kelly_fraction

UNUSUAL_WHALES_URL = "https://api.unusualwhales.com/api/option-trades/flow-alerts"

SYSTEM_PROMPT = """
You are an options-flow analyst. You receive a JSON list of recent large options
trades (whale alerts). Each entry has:
  ticker, strike, expiry, type (call/put), premium_usd, open_interest, sentiment.

Your job:
1. Identify the highest-conviction directional bet.
2. Estimate win probability (0-1) based on flow strength.
3. Return JSON:
{
  "ticker": "...",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "win_prob": 0.0,
  "confidence": 1-10,
  "rationale": "..."
}
"""


def fetch_flow(limit: int = 20, api_key: str = "") -> list[dict]:
    """Pull the latest whale flow from Unusual Whales."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(
            UNUSUAL_WHALES_URL,
            params={"limit": limit},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as exc:
        logger.warning(f"Options flow fetch failed: {exc}")
        return []


def analyze_flow(flow_data: list[dict]) -> dict | None:
    """Ask Claude to interpret the whale flow and return a trade idea."""
    if not flow_data:
        return None
    raw = ask_claude_json(SYSTEM_PROMPT, json.dumps(flow_data[:20]))
    return json.loads(raw)


def execute_flow_trade(idea: dict, capital_usd: float = 500.0) -> None:
    """
    If Claude is bullish and Kelly says go, paper-trade the underlying stock
    (options execution on Alpaca requires a separate options-enabled account).
    """
    if idea.get("direction") != "BULLISH":
        logger.info(f"Skipping non-bullish idea: {idea}")
        return

    win_prob = idea.get("win_prob", 0.5)
    fraction = kelly_fraction(win_prob, win_pct=0.30)
    notional = round(capital_usd * fraction, 2)

    if notional < 10:
        logger.info(f"Kelly fraction too small (${notional}), skipping.")
        return

    logger.info(
        f"Flow trade: {idea['ticker']}  win_prob={win_prob:.0%}  "
        f"kelly={fraction:.2%}  notional=${notional}"
    )
    market_buy(idea["ticker"], notional)


if __name__ == "__main__":
    import os
    api_key = os.getenv("UNUSUAL_WHALES_API_KEY", "")
    flow = fetch_flow(api_key=api_key)
    if not flow:
        # Demo with mock data when no API key present
        flow = [
            {
                "ticker": "SPY", "strike": 500, "expiry": "2025-06-20",
                "type": "call", "premium_usd": 250000,
                "open_interest": 15000, "sentiment": "bullish",
            }
        ]
    idea = analyze_flow(flow)
    print("Flow idea:", json.dumps(idea, indent=2))
