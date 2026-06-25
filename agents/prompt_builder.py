"""
agents/prompt_builder.py -- Prompt template loader and regime-aware selector.

Prompts live outside Python source so they can be versioned, hot-swapped, and
tested independently of code changes.  The autoresearch loop can iterate on
prompt content by writing new version directories without touching any .py file.

Directory layout
----------------
project/prompts/
  pack.json           ← Active pack manifest: regime → (template, version)
  v1/
    trend_following.json
    mean_reversion.json
  v2/                 ← Future versions added here; pack.json switches them in
    ...

Template schema (each JSON file)
---------------------------------
{
  "id":          "trend_following",      # must match filename stem
  "version":     "v1",                   # must match parent directory name
  "strategy":    "trend_following",
  "description": "...",
  "regime_tags": ["trending"],           # informational; selection is via pack
  "text":        "You are a quantitative stock screener..."
}

Pack schema (pack.json)
------------------------
{
  "pack":             "default",
  "version":          "v1",
  "description":      "...",
  "default_template": "trend_following",
  "default_version":  "v1",
  "regime_map": {
    "trending":       {"template": "trend_following", "version": "v1"},
    "mean_reverting": {"template": "mean_reversion",  "version": "v1"},
    "choppy":         {"template": "mean_reversion",  "version": "v1"}
  }
}

Extending / adding a new version
---------------------------------
1.  Create prompts/v2/trend_following.json  (copy v1, edit text)
2.  Update pack.json regime_map entries to "version": "v2"
3.  Call invalidate_cache() if the process is already running
    (watchdog restart also clears the in-process cache automatically)
"""
from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"
PACK_FILE   = PROMPTS_DIR / "pack.json"

# ---------------------------------------------------------------------------
# Template loader  (cached -- templates are immutable once written)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def load_template(template_id: str, version: str = "v1") -> dict:
    """
    Load and cache a prompt template JSON.

    Raises FileNotFoundError if the template doesn't exist so the caller can
    fall back gracefully.
    """
    path = PROMPTS_DIR / version / f"{template_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"prompt_builder | loaded {template_id}/{version}  ({len(data['text'])} chars)")
    return data


def build_prompt(template_id: str, version: str = "v1") -> str:
    """Return the prompt text for the given template ID and version."""
    return load_template(template_id, version)["text"]


# ---------------------------------------------------------------------------
# Pack loader  (cached -- pack changes require invalidate_cache() or restart)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_pack() -> dict:
    """
    Load and cache pack.json.

    Returns an empty dict if the file is missing so all callers fall back to
    the built-in defaults without raising.
    """
    if not PACK_FILE.exists():
        logger.warning(f"prompt_builder | pack.json not found at {PACK_FILE} -- using defaults")
        return {}
    with PACK_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(
        f"prompt_builder | loaded pack '{data.get('pack', '?')}' "
        f"v{data.get('version', '?')}  "
        f"regime_map={list(data.get('regime_map', {}).keys())}"
    )
    return data


# ---------------------------------------------------------------------------
# Cache management  (call after hot-swapping pack.json or adding a new version)
# ---------------------------------------------------------------------------

def invalidate_cache() -> None:
    """Clear all cached templates and the pack manifest."""
    load_template.cache_clear()
    load_pack.cache_clear()
    logger.info("prompt_builder | cache invalidated")


# ---------------------------------------------------------------------------
# Regime-aware prompt selection  (main public API)
# ---------------------------------------------------------------------------

def get_system_prompt(regime=None) -> str:
    """
    Return the appropriate Claude system prompt for the current market regime.

    Selection order:
      1. Consult pack.json regime_map keyed by regime.market_type
      2. Fall back to pack default_template / default_version
      3. Final fallback: trend_following / v1

    Never raises -- every failure path returns a valid (possibly fallback) string.
    """
    pack         = load_pack()
    default_tpl  = pack.get("default_template", "trend_following")
    default_ver  = pack.get("default_version",  "v1")

    template_id = default_tpl
    version     = default_ver

    if regime is not None:
        market_type = getattr(regime, "market_type", None)
        regime_map  = pack.get("regime_map", {})
        if market_type and market_type in regime_map:
            entry       = regime_map[market_type]
            template_id = entry.get("template", default_tpl)
            version     = entry.get("version",  default_ver)
            logger.debug(
                f"prompt_builder | regime={market_type} → {template_id}/{version}"
            )

    try:
        return build_prompt(template_id, version)
    except FileNotFoundError:
        logger.warning(
            f"prompt_builder | template '{template_id}/{version}' missing -- "
            f"falling back to trend_following/v1"
        )

    try:
        return build_prompt("trend_following", "v1")
    except FileNotFoundError:
        logger.error("prompt_builder | trend_following/v1 not found -- returning empty prompt")
        return ""


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def list_available() -> dict[str, list[str]]:
    """Return {version: [template_ids]} for all discovered templates on disk."""
    result: dict[str, list[str]] = {}
    if not PROMPTS_DIR.exists():
        return result
    for version_dir in sorted(PROMPTS_DIR.iterdir()):
        if version_dir.is_dir() and version_dir.name.startswith("v"):
            templates = [p.stem for p in sorted(version_dir.glob("*.json"))]
            if templates:
                result[version_dir.name] = templates
    return result


def active_pack_info() -> dict:
    """Return a summary of the active pack for logging / diagnostics."""
    pack = load_pack()
    return {
        "pack":        pack.get("pack", "default"),
        "version":     pack.get("version", "v1"),
        "regime_map":  pack.get("regime_map", {}),
        "available":   list_available(),
    }
