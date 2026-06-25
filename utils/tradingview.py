"""
utils/tradingview.py -- Optional Python client for the TradingView MCP HTTP server.

Wraps the ngrok-exposed /mcp endpoint so any Python module can call
TradingView tools (chart_set_symbol, tv_health_check, etc.) without
coupling to browser automation.  All calls degrade gracefully:

  - If TV_MCP_URL / MCP_AUTH_TOKEN are not set  -> returns None, logs INFO
  - If the MCP server is unreachable             -> returns None, logs WARNING
  - If TradingView's CDP is disconnected         -> returns None, logs WARNING

Usage:
    from utils.tradingview import tv

    tv.set_symbol("AAPL")           # switch chart to AAPL
    info = tv.health()              # {"cdp_connected": True, ...} or None
    result = tv.call("chart_get_state")
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any
from loguru import logger

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class TradingViewClient:
    """
    Stateless HTTP client for the TradingView MCP server.

    Each call opens a fresh MCP session (POST /mcp with no session-id),
    invokes the tool, and closes.  Intentionally simple -- the TV MCP
    server is local/ngrok so latency is low.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ok: bool | None = None

    # -- Config (read from env each call so hot-reloads work) ------------------

    @staticmethod
    def _cfg() -> tuple[str, str]:
        """Return (url, token) read directly from env vars."""
        url   = os.getenv("TV_MCP_URL", "").rstrip("/")
        token = os.getenv("MCP_AUTH_TOKEN", "")
        return url, token

    def is_configured(self) -> bool:
        url, token = self._cfg()
        return bool(url and token)

    # -- Core MCP call ---------------------------------------------------------

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """
        Call a TradingView MCP tool and return its result value.
        Returns None on any failure so callers never need try/except.
        """
        if not _HAS_REQUESTS:
            logger.debug("tradingview: 'requests' not installed -- skipping.")
            return None

        url, token = self._cfg()
        if not url or not token:
            logger.debug("tradingview: TV_MCP_URL / MCP_AUTH_TOKEN not set -- skipping.")
            return None

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

        # -- Session init ------------------------------------------------------
        init_body = {
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "initialize",
            "params":  {
                "protocolVersion": "2024-11-05",
                "clientInfo":      {"name": "trading-bot-python", "version": "1.0"},
                "capabilities":    {},
            },
        }
        try:
            r = _requests.post(
                f"{url}/mcp",
                headers=headers,
                json=init_body,
                timeout=timeout,
            )
            r.raise_for_status()
            session_id = r.headers.get("mcp-session-id")
            if not session_id:
                logger.warning("tradingview: no mcp-session-id in init response.")
                return None
        except Exception as e:
            if self._last_ok is not False:
                logger.warning(f"tradingview: MCP server unreachable ({e})")
            self._last_ok = False
            return None

        # -- Tool call ---------------------------------------------------------
        call_body = {
            "jsonrpc": "2.0",
            "id":      2,
            "method":  "tools/call",
            "params":  {
                "name":      tool_name,
                "arguments": arguments or {},
            },
        }
        try:
            r2 = _requests.post(
                f"{url}/mcp",
                headers={**headers, "mcp-session-id": session_id},
                json=call_body,
                timeout=timeout,
            )
            r2.raise_for_status()
            data = r2.json()
        except Exception as e:
            logger.warning(f"tradingview: tool call '{tool_name}' failed ({e})")
            return None
        finally:
            # Best-effort session teardown
            try:
                _requests.delete(
                    f"{url}/mcp",
                    headers={**headers, "mcp-session-id": session_id},
                    timeout=3,
                )
            except Exception:
                pass

        # -- Parse result ------------------------------------------------------
        if "error" in data:
            logger.warning(f"tradingview: '{tool_name}' error: {data['error']}")
            return None

        result  = data.get("result", {})
        content = result.get("content", [])
        if not content:
            return result

        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            text = first.get("text", "")
            try:
                parsed = json.loads(text)
                if self._last_ok is not True:
                    logger.debug("tradingview: MCP connection restored.")
                self._last_ok = True
                return parsed
            except json.JSONDecodeError:
                self._last_ok = True
                return text

        self._last_ok = True
        return content

    # -- Convenience wrappers --------------------------------------------------

    def health(self) -> dict | None:
        """Check TradingView MCP + CDP health."""
        return self.call("tv_health_check")

    def is_connected(self) -> bool:
        """True if TV MCP is up AND TradingView's CDP link is live."""
        h = self.health()
        return isinstance(h, dict) and bool(h.get("cdp_connected"))

    def set_symbol(self, ticker: str, exchange: str | None = None) -> bool:
        """Switch the active TradingView chart to ticker."""
        args: dict[str, Any] = {"symbol": ticker.upper()}
        if exchange:
            args["exchange"] = exchange.upper()
        result = self.call("chart_set_symbol", args)
        if result is not None:
            logger.debug(f"tradingview: chart set to {ticker}")
            return True
        return False

    def get_state(self) -> dict | None:
        """Return current chart state (symbol, interval, etc.)."""
        return self.call("chart_get_state")

    def after_trade(self, ticker: str) -> None:
        """
        Best-effort: sync the TV chart to ticker after a trade fires.
        Swallows all errors -- never blocks order execution.
        """
        try:
            self.set_symbol(ticker)
        except Exception:
            pass


# -- Module-level singleton ----------------------------------------------------
tv = TradingViewClient()
