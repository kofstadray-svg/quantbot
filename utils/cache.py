"""
utils/cache.py -- Last-good-data cache for API resilience.

A single API outage (e.g. the Alpaca / yfinance "Max retries exceeded" bursts
seen in the logs) can blind the bot. This module provides a simple on-disk
cache so READ/ANALYSIS data (market data, screening results, watchlists) can
fall back to the last known-good values during an outage, instead of the
screener getting nothing.

IMPORTANT POLICY -- read the design note before using this anywhere new:
  * SAFE to cache:  market data, screen results, watchlists, universe lists.
    Worst case the bot ranks candidates on slightly stale prices / skips a scan.
  * DO NOT cache:   open positions, account balance, fills. Acting on stale
    position data causes phantom orders / double-sells. Anything that places a
    trade must FAIL-CLOSED (do nothing) when live data is unavailable, never
    fall back to cache. See exit_manager / risk for that fail-closed behavior.

Each cache entry stores: the payload, a UTC timestamp, and a label. Callers
decide a max acceptable staleness (max_age_sec) when reading back.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _path(key: str) -> Path:
    # sanitise key into a safe filename
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key).strip("._")
    if not safe:
        safe = "cache_key"

    candidate = _CACHE_DIR / f"{safe}.json"
    cache_root = _CACHE_DIR.resolve()
    resolved = candidate.resolve()
    if resolved.parent != cache_root:
        raise ValueError(f"Invalid cache key path: {key!r}")
    return resolved


def save(key: str, payload, label: str = "") -> None:
    """Write payload as the last-good value for `key`. Never raises (best effort)."""
    try:
        entry = {
            "key": key,
            "label": label or key,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "saved_epoch": time.time(),
            "payload": payload,
        }
        dest = _path(key)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        os.replace(tmp, dest)   # atomic
    except Exception as e:
        logger.debug(f"cache.save({key}) failed (non-fatal): {e}")


def load(key: str, max_age_sec: float | None = None) -> dict | None:
    """
    Return the cached entry dict {payload, saved_at, age_sec, stale} for `key`,
    or None if missing / unreadable / older than max_age_sec.
    """
    p = _path(key)
    if not p.exists():
        return None
    try:
        entry = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"cache.load({key}) unreadable: {e}")
        return None
    age = time.time() - float(entry.get("saved_epoch", 0))
    entry["age_sec"] = age
    entry["stale"] = (max_age_sec is not None and age > max_age_sec)
    if entry["stale"]:
        return None
    return entry


def age_of(key: str) -> float | None:
    """Seconds since `key` was last saved, or None if not cached."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        entry = json.loads(p.read_text(encoding="utf-8"))
        return time.time() - float(entry.get("saved_epoch", 0))
    except Exception:
        return None
