"""
agents/polymarket.py — Day 61-90 of the book.
Scans Polymarket for mispriced binary contracts and sizes bets with Kelly.
"""
from __future__ import annotations
import json
import httpx
from loguru import logger
from utils.claude_client import ask_claude_json
from utils.risk import kelly_fraction, check_position

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"

SYSTEM_PROMPT = """
You are a prediction-market analyst. You receive a list of open Polymarket
markets. Each has: question, current_yes_price (0-1), volume_usd, end_date.

Identify contracts where the current price is significantly mispriced vs your
estimated fair probability.

Respond ONLY with JSON — a list of objects:
[
  {
    "market_id": "...",
    "question": "...",
    "current_price": 0.0,
    "fair_prob": 0.0,
    "edge": 0.0,
    "bet_side": "YES" | "NO",
    "confidence": 1-10,
    "rationale": "..."
  }
]
Return an empty list [] if no edge is found.
"""


def fetch_markets(limit: int = 50) -> list[dict]:
    """Pull active markets from the Polymarket Gamma API (no auth needed)."""
    try:
        r = httpx.get(
            f"{POLYMARKET_GAMMA}/markets",
            params={"active": True, "closed": False, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        markets = r.json()
        return [
            {
                "market_id": m.get("id", ""),
                "question": m.get("question", ""),
                "current_yes_price": float(m.get("outcomePrices", ["0.5"])[0]),
                "volume_usd": float(m.get("volumeNum", 0)),
                "end_date": m.get("endDate", ""),
            }
            for m in markets
        ]
    except Exception as exc:
        logger.error(f"Polymarket fetch failed: {exc}")
        return []


def find_edge(markets: list[dict]) -> list[dict]:
    """Ask Claude to identify mispriced contracts."""
    if not markets:
        return []
    raw = ask_claude_json(SYSTEM_PROMPT, json.dumps(markets))
    ideas = json.loads(raw)
    # Only return trades with meaningful edge and confidence
    return [i for i in ideas if i.get("edge", 0) > 0.05 and i.get("confidence", 0) >= 6]


def size_bets(ideas: list[dict], bankroll_usd: float = 1000.0) -> list[dict]:
    """Apply Kelly criterion to each idea."""
    sized = []
    for idea in ideas:
        fair = idea.get("fair_prob", 0.5)
        price = idea.get("current_price", 0.5)
        # win_pct = payout ratio (binary: win = 1/price - 1)
        if idea["bet_side"] == "YES":
            win_pct = (1 / price) - 1 if price > 0 else 0
        else:
            win_pct = (1 / (1 - price)) - 1 if price < 1 else 0
        fraction = kelly_fraction(fair, win_pct)
        notional = round(bankroll_usd * fraction, 2)
        if notional >= 5 and check_position(notional):
            idea["notional_usd"] = notional
            idea["kelly_fraction"] = round(fraction, 4)
            sized.append(idea)
    return sized


if __name__ == "__main__":
    markets = fetch_markets(limit=30)
    logger.info(f"Fetched {len(markets)} markets")
    ideas = find_edge(markets)
    logger.info(f"Found {len(ideas)} edge opportunities")
    bets = size_bets(ideas)
    for b in bets:
        print(
            f"{b['bet_side']:3s} {b['question'][:60]:60s} "
            f"edge={b['edge']:.2%}  ${b['notional_usd']}"
        )
