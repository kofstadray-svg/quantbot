"""
utils/logging.py — Centralised loguru configuration.

Call setup_logging() once at process startup.  All subsequent
`from loguru import logger` calls in any module will inherit this config.

Format:
  console  — human-readable with colour, level-dependent detail
  file     — JSON records (one per line) for easy grep/jq/log-aggregator use

Usage:
    from utils.logging import setup_logging
    setup_logging(name="bot")     # writes logs/bot_YYYY-MM-DD.jsonl
"""
from __future__ import annotations
import sys
import io
import json
from pathlib import Path
from loguru import logger

# Log directory relative to project root
_LOGS_DIR = Path(__file__).parent.parent / "logs"


def _json_serializer(record: dict) -> str:
    """Serialize a loguru record to a compact JSON line."""
    subset = {
        "ts":      record["time"].isoformat(),
        "level":   record["level"].name,
        "module":  record["name"],
        "func":    record["function"],
        "line":    record["line"],
        "msg":     record["message"],
    }
    # Attach any bound extras (e.g. logger.bind(ticker="AAPL"))
    if record["extra"]:
        subset["extra"] = record["extra"]
    # Attach exception info if present
    if record["exception"]:
        exc = record["exception"]
        subset["exc_type"] = exc.type.__name__ if exc.type else None
        subset["exc_msg"]  = str(exc.value) if exc.value else None
    return json.dumps(subset, default=str)


def _json_format(record: dict) -> str:
    # Escape { and } so loguru's format_map doesn't treat JSON keys as
    # format placeholders.  format_map then unescapes {{ -> { and }} -> }.
    # Escape < and > so loguru's colorizer doesn't treat values like
    # "<module>" as HTML tags (raises ValueError at module level).
    # Loguru unescapes \< -> < on write, so the file content stays correct.
    raw = _json_serializer(record)
    raw = raw.replace("{", "{{").replace("}", "}}")
    raw = raw.replace("<", r"\<").replace(">", r"\>")
    return raw + "\n"


def setup_logging(
    name: str = "app",
    *,
    level: str | None = None,
    retention: str = "30 days",
    rotation: str = "1 day",
) -> None:
    """Configure loguru for the calling process.

    Parameters
    ----------
    name      : label used in the log filename  (e.g. "bot", "scan", "webhook")
    level     : override log level (defaults to DEBUG in dev, INFO in prod)
    retention : how long to keep rotated log files
    rotation  : when to rotate (loguru duration string)
    """
    from config import ENV

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Default level: DEBUG in dev so you can see everything; INFO in prod
    if level is None:
        level = "DEBUG" if ENV == "dev" else "INFO"

    # Remove the default handler loguru adds at import time
    logger.remove()

    # Console sink: coloured, human-readable
    console_fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    # Wrap stderr in a UTF-8 writer that REPLACES un-encodable chars instead of
    # raising. On Windows the console is cp1252; a unicode char (box-drawing,
    # emoji) would otherwise throw UnicodeEncodeError mid-record and silently
    # drop the line that follows. errors="replace" degrades to "?" safely.
    try:
        _console_stream = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace",
            line_buffering=True,
        )
    except Exception:
        _console_stream = sys.stderr  # fallback: stderr has no .buffer (e.g. redirected)
    logger.add(
        _console_stream,
        format=console_fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=(ENV == "dev"),
    )

    # File sink: JSON, one record per line
    log_path = _LOGS_DIR / f"{name}_{{time:YYYY-MM-DD}}.jsonl"
    logger.add(
        str(log_path),
        format=_json_format,
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        serialize=False,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    logger.debug(f"Logging initialised  name={name!r}  level={level}  env={ENV}")
