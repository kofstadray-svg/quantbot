"""
config.py - Central configuration loaded from .env

Call validate_config() at process startup to catch missing/placeholder
values early with a clear error message rather than a cryptic KeyError
deep inside a library.
"""
from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Environment
ENV: str      = os.getenv("ENV", "dev").lower()
IS_PROD: bool = ENV == "prod"

# Claude
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str      = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Alpaca
ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# In dev, always enforce paper trading regardless of the URL in .env
if not IS_PROD:
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
PAPER_TRADING: bool = "paper" in ALPACA_BASE_URL

# Polymarket
_PM_PK: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
_PM_WA: str = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
POLYMARKET_PRIVATE_KEY: str    = _PM_PK
POLYMARKET_WALLET_ADDRESS: str = _PM_WA
POLYMARKET_HOST: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")

POLYMARKET_ENABLED: bool = bool(
    _PM_PK and _PM_WA
    and not _PM_PK.startswith("0x...")
    and not _PM_WA.startswith("0x...")
    and len(_PM_PK) > 10
)

# Tiingo (OHLCV data provider -- replaces yfinance)
TIINGO_API_KEY: str   = os.getenv("TIINGO_API_KEY", "")
TIINGO_ENABLED: bool  = bool(TIINGO_API_KEY)

# Financial Modeling Prep (FMP) — earnings dates, float, RS universe
FMP_API_KEY: str  = os.getenv("FMP_API_KEY", "")
FMP_ENABLED: bool = bool(FMP_API_KEY)

# FinnHub — real-time news, sentiment, earnings surprises, webhook events
FINNHUB_API_KEY: str     = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_ENABLED: bool    = bool(FINNHUB_API_KEY)
FINNHUB_WEBHOOK_SECRET: str = os.getenv("FINNHUB_WEBHOOK_SECRET", "")

# Unusual Whales (options-flow)
UNUSUAL_WHALES_API_KEY: str = os.getenv("UNUSUAL_WHALES_API_KEY", "")
OPTIONS_FLOW_ENABLED: bool  = bool(UNUSUAL_WHALES_API_KEY)

# Telegram
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# Webhook
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_PORT: int   = int(os.getenv("WEBHOOK_PORT", "8080"))

# TradingView MCP
MCP_AUTH_TOKEN: str  = os.getenv("MCP_AUTH_TOKEN", "")
TV_MCP_URL: str      = os.getenv("TV_MCP_URL", "")
TV_MCP_ENABLED: bool = bool(MCP_AUTH_TOKEN and TV_MCP_URL)

# Risk
MAX_POSITION_SIZE_USD: float = float(os.getenv("MAX_POSITION_SIZE_USD", "500"))
DAILY_LOSS_LIMIT_USD: float  = float(os.getenv("DAILY_LOSS_LIMIT_USD", "100"))


# Validation

def _redact_sensitive_values(message: str) -> str:
    """Redact known sensitive config values before logging."""
    redacted = message
    sensitive_values = [
        ANTHROPIC_API_KEY,
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        WEBHOOK_SECRET,
        MCP_AUTH_TOKEN,
        TELEGRAM_BOT_TOKEN,
    ]
    for secret in sensitive_values:
        if secret:
            redacted = redacted.replace(secret, "***REDACTED***")
    return redacted

_REQUIRED: dict[str, str] = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "ALPACA_API_KEY":    ALPACA_API_KEY,
    "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
}

_WEAK_DEFAULTS: dict[str, tuple[str, str]] = {
    "WEBHOOK_SECRET": (WEBHOOK_SECRET, "mysecret123"),
}


def validate_config(*, die: bool = True) -> list[str]:
    """Check required keys are set and warn about known-weak defaults.

    Parameters
    ----------
    die : if True (default), sys.exit(1) on any ERROR-level message.

    Returns a list of message strings (empty = all clear).
    """
    msgs: list[str] = []

    for name, value in _REQUIRED.items():
        if not value:
            msgs.append(f"ERROR: {name} is not set in .env")

    for name, (value, bad) in _WEAK_DEFAULTS.items():
        if value == bad:
            msgs.append(
                f"WARNING: {name} is still the example value '{bad}' -- "
                "change it to a long random string in your .env"
            )

    if not IS_PROD and not PAPER_TRADING:
        msgs.append(
            "ERROR: ENV=dev but ALPACA_BASE_URL points at live trading. "
            "Set ENV=prod to allow live, or leave ENV=dev to enforce paper."
        )
    if IS_PROD and PAPER_TRADING:
        msgs.append(
            "WARNING: ENV=prod but ALPACA_BASE_URL is the paper endpoint. "
            "If intentional (shadow mode), ignore this."
        )

    if not POLYMARKET_ENABLED:
        msgs.append(
            "INFO: Polymarket disabled -- "
            "set real POLYMARKET_PRIVATE_KEY + POLYMARKET_WALLET_ADDRESS to enable."
        )
    if not OPTIONS_FLOW_ENABLED:
        msgs.append(
            "INFO: Options-flow disabled -- set UNUSUAL_WHALES_API_KEY to enable."
        )
    if not TIINGO_ENABLED:
        msgs.append(
            "INFO: Tiingo disabled -- falling back to yfinance for OHLCV. "
            "Set TIINGO_API_KEY to enable the primary data path."
        )
    if not TELEGRAM_BOT_TOKEN:
        msgs.append(
            "INFO: Telegram not configured -- "
            "set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable."
        )
    if not MCP_AUTH_TOKEN:
        msgs.append(
            "WARNING: MCP_AUTH_TOKEN not set -- TradingView ngrok endpoint is unprotected."
        )

    errors = [m for m in msgs if m.startswith("ERROR")]
    if errors and die:
        for m in msgs:
            print(_redact_sensitive_values(m), file=sys.stderr)
        sys.exit(1)

    return msgs
