"""
utils/claude_client.py — Thin wrapper around the Anthropic SDK.
All agents import this instead of instantiating their own clients.
Includes automatic retry with exponential backoff for 529 overload errors.
"""
import time
from anthropic import Anthropic, APIStatusError
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from loguru import logger

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

_MAX_RETRIES = 4          # up to 4 attempts total
_BASE_DELAY  = 5.0        # seconds before first retry
_MAX_DELAY   = 60.0       # cap backoff at 60 s


def ask_claude(system: str, user: str, max_tokens: int = 1024) -> str:
    """Send a single prompt/response to Claude and return the text.
    Retries automatically on 529 overload with exponential backoff."""
    delay = _BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except APIStatusError as e:
            if e.status_code == 529 and attempt < _MAX_RETRIES:
                logger.warning(
                    f"Claude overloaded (529) — retry {attempt}/{_MAX_RETRIES - 1} "
                    f"in {delay:.0f}s…"
                )
                time.sleep(delay)
                delay = min(delay * 2, _MAX_DELAY)
            else:
                raise


def ask_claude_json(system: str, user: str, max_tokens: int = 2048) -> str:
    """Ask Claude for a JSON response. Returns raw JSON string."""
    system_json = system + "\n\nRespond ONLY with valid JSON — no markdown fences."
    return ask_claude(system_json, user, max_tokens)
