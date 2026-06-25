"""
dashboard_server.py — Local web dashboard for the trading bot.

Run:  python dashboard_server.py
URL:  http://localhost:5001

Panels:
  Overview     — service health cards + today's scan schedule
  Activity     — live SSE log stream (bot + telegram + webhook)
  Signals      — parsed BUY/HOLD/SKIP results from today's log
  Watchlist    — add / remove tickers (writes custom_watchlist.py)
  Settings     — safe .env knobs + active prompt pack info
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import queue
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, date
from typing import Iterator

# ── Make sure project root is on sys.path ────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, render_template_string, request

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGS_DIR        = ROOT / "logs"
WATCHLIST_FILE  = ROOT / "data" / "custom_watchlist.py"
ENV_FILE        = ROOT / ".env"
PACK_FILE       = ROOT / "prompts" / "pack.json"

# ── Service health endpoints ──────────────────────────────────────────────────
HEALTH_ENDPOINTS = {
    "bot":     "http://localhost:9102",
    "webhook": "http://localhost:8080",
    "telegram":"http://localhost:9100",
}

# ── SSE subscriber queues ─────────────────────────────────────────────────────
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(16)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today_log(prefix: str) -> pathlib.Path:
    """Return today's JSONL log path for the given prefix (bot/telegram/webhook)."""
    return LOGS_DIR / f"{prefix}_{date.today()}.jsonl"


def _tail_lines(path: pathlib.Path, n: int = 50) -> list[str]:
    """Return the last n lines of a file (fast, no full read for large files)."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]
    except Exception:
        return []


def _parse_jsonl_line(raw: str) -> dict | None:
    """Parse a JSONL log line; return dict or None."""
    try:
        return json.loads(raw)
    except Exception:
        return None


def _health_check(name: str, url: str) -> dict:
    """Hit a /health endpoint. Returns {name, status, latency_ms}."""
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            body = r.read(256).decode()
            ms = round((time.monotonic() - t0) * 1000)
            ok = r.status == 200 and "ready" in body.lower()
            return {"name": name, "status": "up" if ok else "degraded", "latency_ms": ms}
    except Exception:
        ms = round((time.monotonic() - t0) * 1000)
        return {"name": name, "status": "down", "latency_ms": ms}


# ── Signal parsing ────────────────────────────────────────────────────────────
_RE_UNIFIED = re.compile(
    r"Unified\s*\|\s*(\w+)\s+\[([^\]]+)\]\s+conf=([\d.]+)"
)
_RE_SCREEN = re.compile(
    r"Screen\s*\|\s*(\w+)\s+(?:total=([\+\-]?[\d.]+))?.*?->\s*(STRONG BUY|BUY|CONFIRMED|HOLD|WATCH|SKIP|BREAKDOWN|DISABLED)",
    re.IGNORECASE,
)
_RE_SCORE_LINE = re.compile(
    r"(\w{2,6})\s+total=([\+\-]?[\d.]+)\s+\(T=([\+\-]?[\d.]+)\s+M=([\+\-]?[\d.]+)\s+Vt=([\+\-]?[\d.]+)\s+L=([\+\-]?[\d.]+)\s+F=([\+\-]?[\d.]+)\)\s+conf=([\d.]+).*?->\s*(\w+)"
)


def _parse_signals(log_path: pathlib.Path) -> list[dict]:
    """Extract signal results from a bot JSONL log file."""
    signals: dict[str, dict] = {}
    if not log_path.exists():
        return []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                entry = _parse_jsonl_line(raw.strip())
                if not entry:
                    continue
                msg = entry.get("msg", "")
                ts  = entry.get("ts", "")

                # Detailed score line: AAPL total=+2.50 (T=+1.0 M=+0.5 ...) -> BUY
                m = _RE_SCORE_LINE.search(msg)
                if m:
                    ticker = m.group(1).upper()
                    signals[ticker] = {
                        "ticker":     ticker,
                        "total":      float(m.group(2)),
                        "trend":      float(m.group(3)),
                        "momentum":   float(m.group(4)),
                        "volatility": float(m.group(5)),
                        "liquidity":  float(m.group(6)),
                        "fundamental":float(m.group(7)),
                        "confidence": float(m.group(8)),
                        "signal":     m.group(9).upper(),
                        "ts":         ts,
                    }
                    continue

                # Unified screener line
                m = _RE_UNIFIED.search(msg)
                if m:
                    ticker = m.group(1).upper()
                    if ticker not in signals:
                        signals[ticker] = {"ticker": ticker, "ts": ts}
                    signals[ticker]["signal"]     = m.group(2).strip()
                    signals[ticker]["confidence"] = float(m.group(3))
                    continue

                # Simple screen line
                m = _RE_SCREEN.search(msg)
                if m:
                    ticker = m.group(1).upper()
                    if ticker not in signals:
                        signals[ticker] = {"ticker": ticker, "ts": ts}
                    if m.group(2):
                        signals[ticker]["total"] = float(m.group(2))
                    signals[ticker]["signal"] = m.group(3).upper()
    except Exception:
        pass

    result = list(signals.values())
    result.sort(key=lambda x: x.get("total", 0), reverse=True)
    return result


# ── Watchlist helpers ─────────────────────────────────────────────────────────

def _read_watchlist() -> list[str]:
    """Parse CUSTOM_WATCHLIST from custom_watchlist.py."""
    if not WATCHLIST_FILE.exists():
        return []
    try:
        src = WATCHLIST_FILE.read_text(encoding="utf-8")
        m = re.search(r"CUSTOM_WATCHLIST\s*=\s*(\[.*?\])", src, re.DOTALL)
        if not m:
            return []
        return ast.literal_eval(m.group(1))
    except Exception:
        return []


def _write_watchlist(tickers: list[str]) -> None:
    """Rewrite custom_watchlist.py with the new ticker list (rows of 5)."""
    tickers = [t.upper().strip() for t in tickers if t.strip()]
    tickers = sorted(set(tickers))
    rows = []
    for i in range(0, len(tickers), 5):
        chunk = tickers[i:i+5]
        rows.append("    " + ",  ".join(f'"{t}"' for t in chunk) + ",")
    body = "\n".join(rows)
    content = (
        '"""\n'
        'data/custom_watchlist.py — Master custom watchlist.\n'
        'Edit this file to add or remove tickers from the bot\'s stock screener and VWAP scanner.\n'
        '"""\n\n'
        f"CUSTOM_WATCHLIST = [\n{body}\n]\n"
    )
    WATCHLIST_FILE.write_text(content, encoding="utf-8")


# ── Settings helpers ──────────────────────────────────────────────────────────
_SAFE_ENV_KEYS = [
    "ENV", "CLAUDE_MODEL", "WEBHOOK_PORT",
    "MAX_POSITION_SIZE_USD", "DAILY_LOSS_LIMIT_USD",
    "ALPACA_BASE_URL",
]


def _read_env() -> dict[str, str]:
    """Read safe .env keys into a dict."""
    result: dict[str, str] = {}
    if not ENV_FILE.exists():
        return result
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in _SAFE_ENV_KEYS:
            result[key] = val.strip()
    return result


def _write_env_key(key: str, value: str) -> None:
    """Update a single key in .env without touching other lines."""
    if key not in _SAFE_ENV_KEYS:
        raise ValueError(f"Key '{key}' not in safe list")
    if not ENV_FILE.exists():
        ENV_FILE.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    replaced = False
    new_lines = []
    for line in lines:
        if re.match(rf"^{re.escape(key)}\s*=", line):
            new_lines.append(f"{key}={value}\n")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{key}={value}\n")
    ENV_FILE.write_text("".join(new_lines), encoding="utf-8")


# ── Schedule helpers ──────────────────────────────────────────────────────────
# IMPORTANT: This table MUST mirror the schedule.every().day.at(...) calls in
# main.py. Times are machine-local (Mountain). If you change main.py's schedule,
# update this table or the "next" badge will be wrong all day.
# Recurring jobs are omitted: exit/position/risk every 10 min, crypto every 3 hr,
# autoresearch Mon 08:45 / Fri 15:45.
SCHEDULE_ITEMS = [
    {"time": "07:45", "label": "Morning reset — daily loss + regime refresh"},
    {"time": "08:00", "label": "Stock screen — custom watchlist (30 min after open)"},
    {"time": "08:05", "label": "Dow 30 scan"},
    {"time": "08:15", "label": "Nasdaq 100 scan"},
    {"time": "08:25", "label": "Small/mid-cap scan"},
    {"time": "08:40", "label": "VWAP pullback + RSI(2)"},
    {"time": "08:55", "label": "Options flow"},
    {"time": "09:00", "label": "Chart pattern breakouts"},
    {"time": "11:00", "label": "Polymarket"},
    {"time": "11:30", "label": "Midday: stock screen"},
    {"time": "11:40", "label": "Midday: Dow 30"},
    {"time": "11:50", "label": "Midday: Nasdaq 100"},
    {"time": "12:00", "label": "Midday: small/mid-cap"},
    {"time": "13:30", "label": "Pre-close: stock screen (30 min before close)"},
    {"time": "13:32", "label": "Pre-close: VWAP pullback"},
    {"time": "13:35", "label": "Pre-close: chart patterns"},
    {"time": "14:05", "label": "Post-close: cancel unfilled BUYs"},
]


def _schedule_with_status() -> list[dict]:
    now = datetime.now().strftime("%H:%M")
    result = []
    for item in SCHEDULE_ITEMS:
        status = "past" if item["time"] < now else ("next" if not any(
            x["status"] == "next" for x in result) and item["time"] >= now else "upcoming")
        result.append({**item, "status": status})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SSE log tailer
# ─────────────────────────────────────────────────────────────────────────────

def _sse_broadcast(data: str) -> None:
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


def _log_tailer() -> None:
    """Background thread: tail today's log files and push new lines to SSE."""
    prefixes = ["bot", "telegram", "webhook"]
    positions: dict[str, int] = {}
    while True:
        for prefix in prefixes:
            path = _today_log(prefix)
            if not path.exists():
                continue
            pos = positions.get(str(path), 0)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    new_pos = fh.tell()
                if chunk:
                    positions[str(path)] = new_pos
                    for line in chunk.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = _parse_jsonl_line(line)
                        if entry:
                            entry["_source"] = prefix
                            _sse_broadcast(json.dumps(entry))
            except Exception:
                pass
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    # Fan out health probes concurrently so one slow service does not drag
    # the whole endpoint. Was sequential -> ~6s worst case; now ~2s max.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(HEALTH_ENDPOINTS)) as ex:
        futures = {ex.submit(_health_check, name, url): name
                   for name, url in HEALTH_ENDPOINTS.items()}
        results = [f.result() for f in futures]
    # Preserve original key order for the UI
    by_name = {r["name"]: r for r in results}
    ordered = [by_name[n] for n in HEALTH_ENDPOINTS if n in by_name]
    return jsonify(ordered)


@app.route("/api/signals")
def api_signals():
    path = _today_log("bot")
    signals = _parse_signals(path)
    return jsonify({"date": str(date.today()), "signals": signals})


@app.route("/api/logs/backfill")
def api_logs_backfill():
    """Return last 60 lines across all today's logs for initial load."""
    lines = []
    for prefix in ["bot", "telegram", "webhook"]:
        path = _today_log(prefix)
        for raw in _tail_lines(path, 20):
            entry = _parse_jsonl_line(raw)
            if entry:
                entry["_source"] = prefix
                lines.append(entry)
    lines.sort(key=lambda x: x.get("ts", ""))
    return jsonify(lines)


@app.route("/api/logs/stream")
def api_logs_stream():
    """SSE endpoint — push new log lines as they arrive."""
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_subscribers.append(q)

    def generate() -> Iterator[str]:
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    data = q.get(timeout=15)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with _sse_lock:
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/watchlist")
def api_watchlist():
    return jsonify({"tickers": _read_watchlist()})


@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    data = request.get_json(force=True)
    ticker = str(data.get("ticker", "")).upper().strip()
    if not ticker or not re.match(r"^[A-Z]{1,6}$", ticker):
        return jsonify({"error": "invalid ticker"}), 400
    tickers = _read_watchlist()
    if ticker not in tickers:
        tickers.append(ticker)
        _write_watchlist(tickers)
    return jsonify({"tickers": _read_watchlist()})


@app.route("/api/watchlist/remove", methods=["POST"])
def api_watchlist_remove():
    data = request.get_json(force=True)
    ticker = str(data.get("ticker", "")).upper().strip()
    tickers = [t for t in _read_watchlist() if t != ticker]
    _write_watchlist(tickers)
    return jsonify({"tickers": _read_watchlist()})


@app.route("/api/settings")
def api_settings():
    env = _read_env()
    pack: dict = {}
    if PACK_FILE.exists():
        try:
            pack = json.loads(PACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return jsonify({"env": env, "pack": pack})


@app.route("/api/settings", methods=["POST"])
def api_settings_save():
    data = request.get_json(force=True)
    errors = []
    for key, val in data.items():
        try:
            _write_env_key(key, str(val))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400
    return jsonify({"ok": True, "env": _read_env()})


@app.route("/api/schedule")
def api_schedule():
    return jsonify(_schedule_with_status())


# ── Options helpers ───────────────────────────────────────────────────────────
# -- Portfolio API helpers (OpenAlice-style hero metric + curve) --
# Read-only Alpaca queries. No orders. Used by the Portfolio panel.

def _alpaca_client():
    """Return a TradingClient, or None on config / import failure."""
    try:
        from alpaca.trading.client import TradingClient
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
    except Exception:
        return None


@app.route("/api/account")
def api_account():
    """Account summary: equity / cash / buying_power / unrealized + today PnL."""
    client = _alpaca_client()
    if client is None:
        return jsonify({"error": "Alpaca client unavailable"}), 503
    try:
        a = client.get_account()
        positions = client.get_all_positions()
        equity         = float(a.equity)
        last_equity    = float(a.last_equity) if a.last_equity else equity
        cash           = float(a.cash)
        buying_power   = float(a.buying_power)
        unrealized_pl  = sum(float(p.unrealized_pl) for p in positions)
        today_pl       = equity - last_equity
        today_pl_pct   = (today_pl / last_equity * 100.0) if last_equity else 0.0
        return jsonify({
            "equity":         round(equity, 2),
            "last_equity":    round(last_equity, 2),
            "cash":           round(cash, 2),
            "buying_power":   round(buying_power, 2),
            "unrealized_pl":  round(unrealized_pl, 2),
            "today_pl":       round(today_pl, 2),
            "today_pl_pct":   round(today_pl_pct, 3),
            "position_count": len(positions),
            "status":         str(a.status),
            "currency":       "USD",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions")
def api_positions():
    """Open stock + crypto positions. Excludes options (already on Options panel)."""
    client = _alpaca_client()
    if client is None:
        return jsonify({"error": "Alpaca client unavailable", "positions": []}), 503
    try:
        positions = client.get_all_positions()
    except Exception as e:
        return jsonify({"error": str(e), "positions": []}), 500

    out = []
    for p in positions:
        try:
            asset_class = str(getattr(p, "asset_class", "us_equity"))
        except Exception:
            asset_class = "us_equity"
        if "option" in asset_class.lower():
            continue
        qty           = float(p.qty)
        avg_cost      = float(p.avg_entry_price)
        market_price  = float(p.current_price) if p.current_price else 0.0
        market_value  = float(p.market_value) if p.market_value else 0.0
        upl           = float(p.unrealized_pl) if p.unrealized_pl else 0.0
        cost_basis    = abs(avg_cost * qty)
        upl_pct       = (upl / cost_basis * 100.0) if cost_basis else 0.0
        side          = "short" if qty < 0 else "long"
        tag           = "CRYPTO" if "crypto" in asset_class.lower() else "STK"
        out.append({
            "symbol":          p.symbol,
            "tag":             tag,
            "side":            side,
            "qty":             abs(qty),
            "avg_cost":        round(avg_cost, 4),
            "market_price":    round(market_price, 4),
            "market_value":    round(market_value, 2),
            "unrealized_pl":   round(upl, 2),
            "unrealized_plpc": round(upl_pct, 3),
        })
    out.sort(key=lambda r: abs(r["unrealized_pl"]), reverse=True)
    return jsonify({"positions": out, "count": len(out)})


_RANGE_MAP = {
    "1H":  ("1D", "5Min"),
    "6H":  ("1D", "5Min"),
    "24H": ("1D", "5Min"),
    "7D":  ("7D", "15Min"),
    "30D": ("30D", "1H"),
    "ALL": ("all", "1D"),
}


@app.route("/api/equity_curve")
def api_equity_curve():
    """Return [{t, equity}] for the requested range (Alpaca portfolio_history)."""
    rng = (request.args.get("range") or "24H").upper()
    period, timeframe = _RANGE_MAP.get(rng, _RANGE_MAP["24H"])
    client = _alpaca_client()
    if client is None:
        return jsonify({"error": "Alpaca client unavailable", "points": []}), 503
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
        ph  = client.get_portfolio_history(history_filter=req)
        ts_arr = ph.timestamp or []
        eq_arr = ph.equity or []
        points = []
        for t, e in zip(ts_arr, eq_arr):
            if e is None:
                continue
            ev = float(e)
            if ev <= 0:
                # Skip zero/negative equity (Alpaca pads pre-account-creation periods with zeros)
                continue
            points.append({"t": int(t) * 1000, "equity": round(ev, 2)})
        if rng == "1H":
            points = points[-12:]
        elif rng == "6H":
            points = points[-72:]
        return jsonify({"range": rng, "period": period, "timeframe": timeframe, "points": points})
    except Exception as e:
        return jsonify({"error": str(e), "points": []}), 500


_RE_OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def _parse_occ_symbol(sym: str) -> dict:
    """Parse an OCC option symbol like AAPL260717C00335000 into components."""
    m = _RE_OCC.match(sym)
    if not m:
        return {"underlying": sym, "expiry": "", "type": "", "strike": None}
    underlying, yymmdd, cp, strike8 = m.groups()
    expiry = f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    strike = int(strike8) / 1000.0
    return {
        "underlying": underlying,
        "expiry":     expiry,
        "type":       "CALL" if cp == "C" else "PUT",
        "strike":     strike,
    }


@app.route("/api/options")
def api_options():
    """Return open options positions from Alpaca, enriched with parsed OCC fields."""
    try:
        from brokers.alpaca_options import get_options_positions
        positions = get_options_positions()
    except Exception:
        app.logger.exception("Failed to load options positions")
        return jsonify({"error": "Failed to load options positions", "positions": []})

    out = []
    total_value = 0.0
    total_pl = 0.0
    for p in positions:
        meta = _parse_occ_symbol(p.get("symbol", ""))
        mv = p.get("market_value", 0.0) or 0.0
        pl = p.get("unrealized_pl", 0.0) or 0.0
        cb = p.get("cost_basis", 0.0) or 0.0
        pl_pct = (pl / cb * 100.0) if cb else 0.0
        total_value += mv
        total_pl += pl
        out.append({
            "symbol":       p.get("symbol", ""),
            "underlying":   meta["underlying"],
            "type":         meta["type"],
            "strike":       meta["strike"],
            "expiry":       meta["expiry"],
            "qty":          p.get("qty", 0.0),
            "market_value": mv,
            "unrealized_pl": pl,
            "cost_basis":   cb,
            "pl_pct":       pl_pct,
        })
    out.sort(key=lambda x: x.get("unrealized_pl", 0.0), reverse=True)
    return jsonify({
        "positions": out,
        "summary": {
            "count": len(out),
            "total_value": round(total_value, 2),
            "total_pl": round(total_pl, 2),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot Dashboard</title>
<style>
  :root {
    /* GitHub Dark theme - ported from OpenAlice */
    --bg-1:           #0d1117;
    --bg-2:           #161b22;
    --bg-3:           #21262d;
    --bg-4:           #30363d;
    --border:         #30363d;
    --text:           #e6edf3;
    --muted:          #8b949e;
    --green:          #3fb950;
    --red:            #f85149;
    --yellow:         #d29922;
    --blue:           #58a6ff;
    --purple:         #a78bfa;
    /* OpenAlice naming aliases (Portfolio panel uses these) */
    --bg:             var(--bg-1);
    --bg-secondary:   var(--bg-2);
    --bg-tertiary:    var(--bg-3);
    --text-muted:     var(--muted);
    --accent:         #58a6ff;
    --accent-dim:     rgba(31, 111, 235, 0.2);
    --radius:         6px;
    --font:           -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono:      'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg-1); color: var(--text); font-family: var(--font); font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* ── Header ── */
  header { background: var(--bg-2); border-bottom: 1px solid var(--border); padding: 0 16px; display: flex; align-items: center; gap: 20px; height: 44px; flex-shrink: 0; }
  .logo { font-size: 15px; font-weight: 700; color: var(--green); letter-spacing: .5px; }
  .logo span { color: var(--text); font-weight: 400; }
  nav { display: flex; gap: 2px; }
  nav button { background: none; border: none; color: var(--muted); padding: 6px 14px; border-radius: var(--radius); cursor: pointer; font-size: 13px; transition: .15s; }
  nav button:hover { color: var(--text); background: var(--bg-3); }
  nav button.active { color: var(--text); background: var(--bg-3); }
  .header-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
  .clock { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .refresh-btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--text); padding: 4px 12px; border-radius: var(--radius); cursor: pointer; font-size: 12px; }
  .refresh-btn:hover { background: var(--bg-4); }

  /* ── Layout ── */
  main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .panel { display: none; flex: 1; overflow-y: auto; padding: 16px; }
  .panel.active { display: flex; flex-direction: column; gap: 14px; }

  /* ── Cards ── */
  .card { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); }
  .card-header { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); display: flex; align-items: center; justify-content: space-between; }
  .card-body { padding: 12px 14px; }

  /* ── Status cards ── */
  .status-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .svc-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; display: flex; align-items: center; gap: 12px; }
  .svc-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .svc-dot.up { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .svc-dot.down { background: var(--red); box-shadow: 0 0 6px var(--red); }
  .svc-dot.degraded { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
  .svc-dot.unknown { background: var(--muted); }
  .svc-name { font-weight: 600; font-size: 13px; }
  .svc-meta { color: var(--muted); font-size: 11px; margin-top: 1px; }

  /* ── Schedule ── */
  .schedule-list { display: flex; flex-direction: column; gap: 4px; }
  .sched-item { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--bg-3); }
  .sched-item:last-child { border-bottom: none; }
  .sched-time { width: 50px; font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; }
  .sched-label { flex: 1; }
  .sched-badge { font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; }
  .sched-badge.past { background: var(--bg-3); color: var(--muted); }
  .sched-badge.next { background: var(--green); color: #000; }
  .sched-badge.upcoming { background: var(--bg-3); color: var(--text); }

  /* ── Activity feed ── */
  #log-feed { flex: 1; overflow-y: auto; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11.5px; line-height: 1.55; padding: 8px 12px; background: var(--bg-1); border-radius: var(--radius); min-height: 0; }
  .log-line { display: flex; gap: 8px; padding: 1px 0; }
  .log-line:hover { background: var(--bg-2); }
  .log-ts { color: var(--muted); flex-shrink: 0; }
  .log-src { width: 60px; flex-shrink: 0; font-size: 10px; padding: 1px 5px; border-radius: 3px; text-align: center; align-self: center; }
  .log-src.bot { background: #1a237e; color: #90caf9; }
  .log-src.telegram { background: #1a3a2a; color: #80cbc4; }
  .log-src.webhook { background: #3e2723; color: #ffcc80; }
  .log-level { width: 48px; flex-shrink: 0; font-size: 10px; text-align: right; }
  .log-level.DEBUG { color: var(--muted); }
  .log-level.INFO { color: var(--blue); }
  .log-level.WARNING { color: var(--yellow); }
  .log-level.ERROR, .log-level.CRITICAL { color: var(--red); }
  .log-msg { flex: 1; word-break: break-all; color: var(--text); }
  .log-controls { display: flex; gap: 8px; align-items: center; }
  .log-controls label { color: var(--muted); font-size: 12px; display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .filter-btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--muted); padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .filter-btn.active { border-color: var(--blue); color: var(--blue); }

  /* ── Signals ── */
  .signals-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
  .sig-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; cursor: default; transition: border-color .15s; }
  .sig-card:hover { border-color: var(--bg-4); }
  .sig-card.BUY, .sig-card.STRONG_BUY, .sig-card.CONFIRMED { border-left: 3px solid var(--green); }
  .sig-card.HOLD, .sig-card.WATCH { border-left: 3px solid var(--yellow); }
  .sig-card.SKIP, .sig-card.BREAKDOWN { border-left: 3px solid var(--red); }
  .sig-ticker { font-size: 16px; font-weight: 700; letter-spacing: .5px; }
  .sig-signal { font-size: 11px; font-weight: 600; margin-top: 1px; }
  .sig-signal.buy { color: var(--green); }
  .sig-signal.hold { color: var(--yellow); }
  .sig-signal.skip { color: var(--red); }
  .sig-bars { display: flex; gap: 4px; margin-top: 8px; }
  .sig-bar-wrap { flex: 1; text-align: center; }
  .sig-bar-label { font-size: 9px; color: var(--muted); margin-bottom: 2px; }
  .sig-bar { height: 32px; border-radius: 2px; position: relative; display: flex; align-items: flex-end; justify-content: center; font-size: 9px; }
  .sig-conf { margin-top: 6px; font-size: 10px; color: var(--muted); }
  .sig-conf span { color: var(--text); }
  .signals-empty { color: var(--muted); text-align: center; padding: 40px 0; font-size: 14px; }
  .sig-filters { display: flex; gap: 6px; flex-wrap: wrap; }
  .sig-filter { background: var(--bg-3); border: 1px solid var(--border); color: var(--muted); padding: 4px 12px; border-radius: 20px; cursor: pointer; font-size: 11px; }
  .sig-filter.active { border-color: var(--green); color: var(--green); }

  /* ── Options ── */
  .opt-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .opt-stat { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; }
  .opt-stat-val { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .opt-stat-val.pos { color: var(--green); }
  .opt-stat-val.neg { color: var(--red); }
  .opt-stat-label { color: var(--muted); font-size: 11px; margin-top: 2px; text-transform: uppercase; letter-spacing: .5px; }
  .opt-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  .opt-table th { text-align: left; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .6px; padding: 6px 10px; border-bottom: 1px solid var(--border); font-weight: 600; }
  .opt-table th.num, .opt-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .opt-table td { padding: 8px 10px; border-bottom: 1px solid var(--bg-3); }
  .opt-table tr:hover td { background: var(--bg-2); }
  .opt-table .occ { font-family: 'Cascadia Code','Consolas',monospace; font-size: 11px; color: var(--muted); }
  .opt-table .under { font-weight: 700; font-size: 13px; }
  .opt-type { font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; }
  .opt-type.CALL { background: rgba(38,166,154,.2); color: var(--green); }
  .opt-type.PUT { background: rgba(239,83,80,.2); color: var(--red); }
  .opt-pl.pos { color: var(--green); }
  .opt-pl.neg { color: var(--red); }
  .opt-empty { color: var(--muted); text-align: center; padding: 40px 0; }

  /* ── Watchlist ── */
  .ticker-grid { display: flex; flex-wrap: wrap; gap: 6px; }
  .ticker-chip { background: var(--bg-3); border: 1px solid var(--border); border-radius: 4px; padding: 5px 10px; display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; }
  .ticker-chip button { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; line-height: 1; padding: 0; }
  .ticker-chip button:hover { color: var(--red); }
  .add-ticker { display: flex; gap: 8px; align-items: center; }
  .add-ticker input { background: var(--bg-3); border: 1px solid var(--border); color: var(--text); padding: 7px 12px; border-radius: var(--radius); font-size: 13px; width: 140px; outline: none; text-transform: uppercase; }
  .add-ticker input:focus { border-color: var(--blue); }
  .btn-primary { background: var(--green); border: none; color: #000; padding: 7px 16px; border-radius: var(--radius); cursor: pointer; font-size: 13px; font-weight: 600; }
  .btn-primary:hover { opacity: .85; }
  .wl-count { color: var(--muted); font-size: 11px; }

  /* ── Settings ── */
  .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .setting-row { display: flex; flex-direction: column; gap: 5px; }
  .setting-row label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .setting-row input, .setting-row select { background: var(--bg-3); border: 1px solid var(--border); color: var(--text); padding: 7px 10px; border-radius: var(--radius); font-size: 13px; outline: none; }
  .setting-row input:focus, .setting-row select:focus { border-color: var(--blue); }
  .settings-save { display: flex; gap: 10px; align-items: center; margin-top: 4px; }
  .save-msg { font-size: 12px; color: var(--green); opacity: 0; transition: opacity .3s; }
  .save-msg.show { opacity: 1; }
  .pack-info { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12px; }
  .pack-key { color: var(--muted); }
  .pack-val { color: var(--text); font-family: monospace; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-1); }
  ::-webkit-scrollbar-thumb { background: var(--bg-4); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

  /* ── Misc ── */
  .row { display: flex; gap: 12px; }
  .row > * { flex: 1; }
  .tag { font-size: 10px; padding: 2px 6px; border-radius: 3px; }
  .tag.green { background: rgba(38,166,154,.2); color: var(--green); }
  .tag.red { background: rgba(239,83,80,.2); color: var(--red); }
  .tag.yellow { background: rgba(245,159,0,.2); color: var(--yellow); }
  .tag.blue { background: rgba(33,150,243,.2); color: var(--blue); }

  /* ============================================
     OpenAlice-ported styles — Portfolio panel
     ============================================ */

  /* Live pulse dot — used in the brand */
  @keyframes live-pulse-anim {
    0%, 100% { opacity: 0.55; transform: scale(0.92); }
    50%      { opacity: 1;    transform: scale(1.08); }
  }
  .live-pulse {
    position: relative;
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: live-pulse-anim 2.4s ease-in-out infinite;
  }
  .live-pulse::after {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 9999px;
    background: var(--green);
    opacity: 0.3;
    animation: live-pulse-anim 2.4s ease-in-out infinite;
    animation-delay: 0.3s;
  }

  /* Hero metric block — Total Equity centerpiece */
  .hero-metric {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
  }
  .metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
  }
  .metric-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
    color: var(--text);
  }
  .metric-value.sm    { font-size: 18px; font-weight: 600; }
  .metric-value.up    { color: var(--green); }
  .metric-value.down  { color: var(--red); }
  .metric-delta {
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    margin-top: 4px;
    color: var(--text-muted);
  }
  .metric-delta.up   { color: var(--green); }
  .metric-delta.down { color: var(--red); }
  .metric-secondary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }

  /* Equity curve card */
  .equity-card { padding: 16px; }
  .equity-card .card-header { padding: 0 0 12px 0; border-bottom: none; }
  .range-pills { display: flex; gap: 2px; }
  .range-pill {
    padding: 4px 10px;
    font-size: 11px;
    border-radius: 4px;
    background: transparent;
    color: var(--text-muted);
    border: none;
    cursor: pointer;
    transition: 0.15s;
    font-variant-numeric: tabular-nums;
  }
  .range-pill:hover  { background: var(--bg-tertiary); color: var(--text); }
  .range-pill.active { background: var(--accent-dim); color: var(--accent); font-weight: 600; }
  #equity-chart-wrap { height: 240px; position: relative; }
  #equity-chart-empty {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-muted); font-size: 13px;
  }

  /* Positions table - OpenAlice style */
  .pos-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .pos-table th {
    text-align: left;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }
  .pos-table th.num { text-align: right; }
  .pos-table td {
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    transition: background 0.1s;
  }
  .pos-table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .pos-table tbody tr:hover td { background: rgba(33, 38, 45, 0.5); }
  .pos-table .sym-name { font-weight: 500; color: var(--text); }
  .pos-table .sym-tag {
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-left: 6px;
    letter-spacing: 0;
  }
  .pos-table .sym-tag.short {
    background: rgba(248, 81, 73, 0.15);
    color: var(--red);
  }
  .pos-table .pos-pl.pos  { color: var(--green); font-weight: 500; }
  .pos-table .pos-pl.neg  { color: var(--red);   font-weight: 500; }
  .pos-empty {
    color: var(--text-muted);
    text-align: center;
    padding: 30px 0;
    font-size: 13px;
  }


  /* ============================================
     Polish pass — existing panels feel cohesive
     ============================================ */

  /* Unify section labels (h3/uppercase pattern from OpenAlice) */
  h3.section-label, .section-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin: 4px 0 10px 0;
  }

  /* Service health cards — tighter, cleaner dot (no glow) */
  .svc-card {
    padding: 12px 16px;
    gap: 10px;
  }
  .svc-dot {
    width: 8px;
    height: 8px;
    box-shadow: none;
  }
  .svc-dot.up { background: var(--green); }
  .svc-dot.down { background: var(--red); }
  .svc-dot.degraded { background: var(--yellow); }
  .svc-name { font-weight: 500; }
  .svc-meta {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
    font-variant-numeric: tabular-nums;
  }

  /* Schedule list — refined spacing */
  .sched-item {
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
  }
  .sched-time {
    width: 52px;
    color: var(--text-muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .sched-label { font-size: 13px; color: var(--text); }
  .sched-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .sched-badge.past { background: var(--bg-tertiary); color: var(--text-muted); }
  .sched-badge.next { background: var(--accent); color: var(--bg); }
  .sched-badge.upcoming { background: var(--bg-tertiary); color: var(--text); }

  /* Signal cards — match OpenAlice card pattern */
  .sig-card {
    padding: 12px 14px;
    background: var(--bg-secondary);
    transition: border-color 0.15s;
  }
  .sig-card:hover { border-color: var(--bg-tertiary); }
  .sig-ticker { font-size: 15px; font-weight: 600; letter-spacing: 0; }
  .sig-signal { font-size: 10px; font-weight: 600; letter-spacing: 0.04em; }
  .sig-conf {
    font-size: 11px;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
  }
  .sig-filter {
    padding: 4px 12px;
    border-radius: var(--radius);
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
  }
  .sig-filter:hover { color: var(--text); background: var(--bg-tertiary); }
  .sig-filter.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: transparent;
    font-weight: 500;
  }

  /* Options stat cards — match hero-metric secondary pattern */
  .opt-stat {
    padding: 14px 16px;
  }
  .opt-stat-val {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.1;
  }
  .opt-stat-val.pos { color: var(--green); }
  .opt-stat-val.neg { color: var(--red); }
  .opt-stat-label {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
  }

  /* Options table — match positions table */
  .opt-table { font-size: 13px; }
  .opt-table th {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 500;
    color: var(--text-muted);
    padding: 8px 12px;
  }
  .opt-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .opt-table tr:last-child td { border-bottom: none; }
  .opt-table .occ { color: var(--text-muted); font-family: var(--font-mono); }
  .opt-type {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 3px;
    font-weight: 600;
    font-family: var(--font-mono);
    letter-spacing: 0;
  }

  /* Watchlist chips — smaller, more refined */
  .ticker-chip {
    background: var(--bg-tertiary);
    border-color: var(--border);
    font-size: 12px;
    font-weight: 600;
    padding: 4px 4px 4px 10px;
  }
  .ticker-chip button:hover { color: var(--red); }

  /* Buttons — unified btn pattern alongside legacy classes */
  .btn-primary {
    background: var(--accent);
    color: var(--bg);
    font-weight: 600;
  }
  .btn-primary:hover { opacity: 0.9; }
  .refresh-btn, .filter-btn, .sig-filter {
    font-family: var(--font);
  }
  .refresh-btn {
    background: var(--bg-tertiary);
    border-color: var(--border);
    color: var(--text);
    transition: 0.15s;
  }
  .refresh-btn:hover {
    border-color: var(--accent);
    background: var(--bg-tertiary);
  }

  /* Settings form polish */
  .setting-row label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
  }
  .setting-row input, .setting-row select {
    background: var(--bg-tertiary);
    border-color: var(--border);
    font-size: 13px;
    padding: 7px 10px;
  }
  .setting-row input:focus, .setting-row select:focus {
    border-color: var(--accent);
  }
  .pack-info { font-size: 12px; line-height: 1.7; }
  .pack-key { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  .pack-val { color: var(--text); font-family: var(--font-mono); font-size: 12px; }

  /* ============================================
     Portfolio sub-widgets (top movers / mix / risk)
     ============================================ */

  .widget-strip {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 12px;
  }
  @media (max-width: 1100px) {
    .widget-strip { grid-template-columns: 1fr; }
  }

  /* Top movers */
  .movers-card {
    padding: 14px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .movers-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 8px;
  }
  .movers-col {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .movers-col-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 2px;
  }
  .mover-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 0;
    font-size: 12px;
  }
  .mover-sym {
    color: var(--text);
    font-weight: 500;
    font-size: 13px;
    flex-shrink: 0;
  }
  .mover-pct {
    font-variant-numeric: tabular-nums;
    font-weight: 500;
    font-size: 12px;
    text-align: right;
  }
  .mover-pct.pos { color: var(--green); }
  .mover-pct.neg { color: var(--red); }
  .mover-empty {
    color: var(--text-muted);
    font-size: 11px;
    padding: 4px 0;
  }

  /* Asset mix */
  .mix-card, .risk-card {
    padding: 14px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .mix-row, .risk-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 13px;
  }
  .mix-row + .mix-row, .risk-row + .risk-row { border-top: 1px solid var(--border); }
  .mix-label, .risk-label {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .mix-val, .risk-val {
    color: var(--text);
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }
  .mix-bar-wrap {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
    display: flex;
  }
  .mix-bar-seg { height: 100%; }
  .mix-bar-seg.stk    { background: var(--accent); }
  .mix-bar-seg.crypto { background: var(--purple); }
  .mix-bar-seg.cash   { background: var(--bg-4); }

</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <div class="logo"><span class="live-pulse" style="margin-right: 10px;"></span>⚡ Bot<span>Dashboard</span></div>
  <nav>
    <button class="active" onclick="showPanel('portfolio', this)">Portfolio</button>
    <button onclick="showPanel('overview', this)">Overview</button>
    <button onclick="showPanel('activity', this)">Activity</button>
    <button onclick="showPanel('signals', this)">Signals</button>
    <button onclick="showPanel('options', this)">Options</button>
    <button onclick="showPanel('watchlist', this)">Watchlist</button>
    <button onclick="showPanel('settings', this)">Settings</button>
  </nav>
  <div class="header-right">
    <div id="last-updated" style="color: var(--text-muted); font-size: 11px; margin-right: 6px;">—</div>
    <div class="clock" id="clock">--:--:--</div>
    <button class="refresh-btn" onclick="refreshAll()">↻ Refresh</button>
  </div>
</header>

<main>


<!-- PORTFOLIO -->
<div class="panel active" id="panel-portfolio">

  <!-- Hero metric block -->
  <div class="hero-metric">
    <div class="metric-label">Total Equity &middot; USD</div>
    <div class="metric-value" id="hero-equity">&mdash;</div>
    <div class="metric-delta" id="hero-delta">&mdash; today</div>
    <div class="metric-secondary">
      <div>
        <div class="metric-label">Cash</div>
        <div class="metric-value sm" id="hero-cash">&mdash;</div>
      </div>
      <div>
        <div class="metric-label">Unrealized PnL</div>
        <div class="metric-value sm" id="hero-upl">&mdash;</div>
      </div>
      <div>
        <div class="metric-label">Buying Power</div>
        <div class="metric-value sm" id="hero-bp">&mdash;</div>
      </div>
    </div>
  </div>


  <!-- Sub-widgets strip: top movers / asset mix / risk -->
  <div class="widget-strip">
    <div class="movers-card">
      <div class="section-label" style="margin: 0 0 4px 0;">Top Movers</div>
      <div class="movers-grid">
        <div class="movers-col">
          <div class="movers-col-label">Gainers</div>
          <div id="movers-up"><div class="mover-empty">—</div></div>
        </div>
        <div class="movers-col">
          <div class="movers-col-label">Losers</div>
          <div id="movers-down"><div class="mover-empty">—</div></div>
        </div>
      </div>
    </div>

    <div class="mix-card">
      <div class="section-label" style="margin: 0 0 8px 0;">Asset Mix</div>
      <div id="mix-rows">
        <div class="mix-row"><span class="mix-label">Loading…</span></div>
      </div>
      <div class="mix-bar-wrap" id="mix-bar"></div>
    </div>

    <div class="risk-card">
      <div class="section-label" style="margin: 0 0 8px 0;">Risk</div>
      <div id="risk-rows">
        <div class="risk-row"><span class="risk-label">Loading…</span></div>
      </div>
    </div>
  </div>

  <!-- Equity curve -->
  <div class="card equity-card">
    <div class="card-header">
      <span>Equity Curve</span>
      <div class="range-pills" id="range-pills">
        <button class="range-pill" data-rng="1H"  onclick="setRange('1H', this)">1H</button>
        <button class="range-pill" data-rng="6H"  onclick="setRange('6H', this)">6H</button>
        <button class="range-pill active" data-rng="24H" onclick="setRange('24H', this)">24H</button>
        <button class="range-pill" data-rng="7D"  onclick="setRange('7D', this)">7D</button>
        <button class="range-pill" data-rng="30D" onclick="setRange('30D', this)">30D</button>
        <button class="range-pill" data-rng="All" onclick="setRange('All', this)">All</button>
      </div>
    </div>
    <div id="equity-chart-wrap">
      <canvas id="equity-chart"></canvas>
      <div id="equity-chart-empty" style="display:none;">Loading equity curve&hellip;</div>
    </div>
  </div>

  <!-- Positions -->
  <div>
    <h3 class="metric-label" style="margin: 4px 0 8px 0;">Positions</h3>
    <div class="card" style="overflow-x: auto;">
      <table class="pos-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th class="num">Qty</th>
            <th class="num">Avg Cost</th>
            <th class="num">Current</th>
            <th class="num">Market Value</th>
            <th class="num">PnL</th>
            <th class="num">PnL %</th>
          </tr>
        </thead>
        <tbody id="positions-tbody">
          <tr><td colspan="7" class="pos-empty">Loading positions&hellip;</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- OVERVIEW -->
<div class="panel" id="panel-overview">
  <div class="status-grid" id="status-grid">
    <div class="svc-card"><div class="svc-dot unknown"></div><div><div class="svc-name">Bot</div><div class="svc-meta">checking…</div></div></div>
    <div class="svc-card"><div class="svc-dot unknown"></div><div><div class="svc-name">Webhook</div><div class="svc-meta">checking…</div></div></div>
    <div class="svc-card"><div class="svc-dot unknown"></div><div><div class="svc-name">Telegram</div><div class="svc-meta">checking…</div></div></div>
  </div>
  <div class="row">
    <div class="card">
      <div class="card-header">Today's Scan Schedule</div>
      <div class="card-body"><div class="schedule-list" id="schedule-list">Loading…</div></div>
    </div>
    <div class="card">
      <div class="card-header">Today at a Glance</div>
      <div class="card-body" id="glance-body">Loading…</div>
    </div>
  </div>
</div>

<!-- ACTIVITY -->
<div class="panel" id="panel-activity" style="padding-bottom:0;">
  <div style="display:flex; gap:8px; align-items:center; flex-shrink:0; padding-bottom:8px;">
    <span style="color:var(--muted);font-size:11px;">Filter:</span>
    <button class="filter-btn active" onclick="toggleFilter('bot',this)">Bot</button>
    <button class="filter-btn active" onclick="toggleFilter('telegram',this)">Telegram</button>
    <button class="filter-btn active" onclick="toggleFilter('webhook',this)">Webhook</button>
    <span style="margin-left:8px; color:var(--muted); font-size:11px;">Level:</span>
    <button class="filter-btn" onclick="toggleLevel('DEBUG',this)">DEBUG</button>
    <button class="filter-btn active" onclick="toggleLevel('INFO',this)">INFO+</button>
    <label style="margin-left:auto; color:var(--muted); font-size:12px; cursor:pointer;">
      <input type="checkbox" id="autoscroll" checked> Auto-scroll
    </label>
    <button class="filter-btn" onclick="clearFeed()">Clear</button>
    <span style="color:var(--muted);font-size:11px;" id="log-count">0 lines</span>
  </div>
  <div id="log-feed"></div>
</div>

<!-- SIGNALS -->
<div class="panel" id="panel-signals">
  <div class="card">
    <div class="card-header">
      <span>Screener Results — <span id="sig-date">today</span></span>
      <div class="sig-filters">
        <div class="sig-filter active" onclick="setSigFilter('ALL',this)">All</div>
        <div class="sig-filter" onclick="setSigFilter('BUY',this)">Buy</div>
        <div class="sig-filter" onclick="setSigFilter('HOLD',this)">Hold</div>
        <div class="sig-filter" onclick="setSigFilter('SKIP',this)">Skip</div>
      </div>
    </div>
    <div class="card-body">
      <div class="signals-grid" id="signals-grid"><div class="signals-empty">No signals found for today yet.</div></div>
    </div>
  </div>
</div>

<!-- OPTIONS -->
<div class="panel" id="panel-options">
  <div class="opt-summary" id="opt-summary">
    <div class="opt-stat"><div class="opt-stat-val" id="opt-count">—</div><div class="opt-stat-label">open positions</div></div>
    <div class="opt-stat"><div class="opt-stat-val" id="opt-value">—</div><div class="opt-stat-label">market value</div></div>
    <div class="opt-stat"><div class="opt-stat-val" id="opt-pl">—</div><div class="opt-stat-label">unrealized P/L</div></div>
  </div>
  <div class="card">
    <div class="card-header"><span>Open Options Positions</span></div>
    <div class="card-body">
      <table class="opt-table" id="opt-table">
        <thead>
          <tr><th>Contract</th><th>Type</th><th>Strike</th><th>Expiry</th><th class="num">Qty</th><th class="num">Value</th><th class="num">Unreal. P/L</th><th class="num">%</th></tr>
        </thead>
        <tbody id="opt-tbody"><tr><td colspan="8" class="opt-empty">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<!-- WATCHLIST -->
<div class="panel" id="panel-watchlist">
  <div class="card">
    <div class="card-header">
      <span>Custom Watchlist <span class="wl-count" id="wl-count"></span></span>
      <div class="add-ticker">
        <input type="text" id="ticker-input" placeholder="TICKER" maxlength="6"
               onkeydown="if(event.key==='Enter') addTicker()">
        <button class="btn-primary" onclick="addTicker()">+ Add</button>
      </div>
    </div>
    <div class="card-body">
      <div class="ticker-grid" id="ticker-grid">Loading…</div>
    </div>
  </div>
</div>

<!-- SETTINGS -->
<div class="panel" id="panel-settings">
  <div class="row">
    <div class="card">
      <div class="card-header">Bot Settings (.env)</div>
      <div class="card-body" id="settings-body">Loading…</div>
    </div>
    <div class="card">
      <div class="card-header">Active Prompt Pack</div>
      <div class="card-body"><div class="pack-info" id="pack-info">Loading…</div></div>
    </div>
  </div>
</div>

</main>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const filters = { bot: true, telegram: true, webhook: true };
const levelMin = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 };
let currentLevel = 1;  // INFO+
let sigFilter = 'ALL';
let allSignals = [];
let logLineCount = 0;
const MAX_LOG_LINES = 500;

// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ── Panel nav ─────────────────────────────────────────────────────────────
function showPanel(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'portfolio'){ loadPortfolio(); }
  if (name === 'overview') { loadStatus(); loadSchedule(); loadGlance(); }
  if (name === 'signals')  { loadSignals(); }
  if (name === 'options')  { loadOptions(); }
  if (name === 'watchlist'){ loadWatchlist(); }
  if (name === 'settings') { loadSettings(); }
}

// ── Status ─────────────────────────────────────────────────────────────────
async function loadStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  const grid = document.getElementById('status-grid');
  grid.innerHTML = data.map(s => `
    <div class="svc-card">
      <div class="svc-dot ${s.status}"></div>
      <div>
        <div class="svc-name">${capitalize(s.name)}</div>
        <div class="svc-meta">${s.status} · ${s.latency_ms}ms</div>
      </div>
    </div>`).join('');
}

// ── Schedule ───────────────────────────────────────────────────────────────
async function loadSchedule() {
  const res = await fetch('/api/schedule');
  const items = await res.json();
  document.getElementById('schedule-list').innerHTML = items.map(i => `
    <div class="sched-item">
      <div class="sched-time">${i.time}</div>
      <div class="sched-label">${i.label}</div>
      <div class="sched-badge ${i.status}">${i.status === 'next' ? '▶ next' : i.status}</div>
    </div>`).join('');
}

// ── Glance ─────────────────────────────────────────────────────────────────
async function loadGlance() {
  const res = await fetch('/api/signals');
  const data = await res.json();
  const sigs = data.signals;
  const buys = sigs.filter(s => /BUY|CONFIRM/i.test(s.signal || '')).length;
  const holds = sigs.filter(s => /HOLD|WATCH/i.test(s.signal || '')).length;
  const skips = sigs.filter(s => /SKIP|BREAK/i.test(s.signal || '')).length;
  document.getElementById('glance-body').innerHTML = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;">
      <div><div style="font-size:28px;font-weight:700;color:var(--green)">${buys}</div><div style="color:var(--muted);font-size:11px">BUY signals</div></div>
      <div><div style="font-size:28px;font-weight:700;color:var(--yellow)">${holds}</div><div style="color:var(--muted);font-size:11px">HOLD</div></div>
      <div><div style="font-size:28px;font-weight:700;color:var(--muted)">${skips}</div><div style="color:var(--muted);font-size:11px">SKIP</div></div>
      <div><div style="font-size:28px;font-weight:700;color:var(--text)">${sigs.length}</div><div style="color:var(--muted);font-size:11px">total screened</div></div>
    </div>`;
}

// ── Activity / SSE ─────────────────────────────────────────────────────────
function toggleFilter(src, btn) {
  filters[src] = !filters[src];
  btn.classList.toggle('active', filters[src]);
  renderLogFeed();
}
function toggleLevel(lvl, btn) {
  if (lvl === 'DEBUG') {
    currentLevel = currentLevel === 0 ? 1 : 0;
    btn.classList.toggle('active', currentLevel === 0);
    document.querySelector('.filter-btn[onclick*="INFO"]').classList.toggle('active', currentLevel <= 1);
  } else {
    currentLevel = 1;
    btn.classList.add('active');
    document.querySelector('.filter-btn[onclick*="DEBUG"]').classList.remove('active');
  }
  renderLogFeed();
}
function clearFeed() { window._logLines = []; renderLogFeed(); }

window._logLines = [];
function addLogLine(entry) {
  window._logLines.push(entry);
  if (window._logLines.length > MAX_LOG_LINES) window._logLines.shift();
  logLineCount++;
}
function renderLogFeed() {
  const feed = document.getElementById('log-feed');
  const lvlOrder = { DEBUG:0, INFO:1, WARNING:2, ERROR:3, CRITICAL:4 };
  const lines = window._logLines.filter(e => {
    if (!filters[e._source]) return false;
    if ((lvlOrder[e.level] ?? 0) < currentLevel) return false;
    return true;
  });
  feed.innerHTML = lines.map(e => {
    const ts = e.ts ? e.ts.substr(11,8) : '';
    const lvl = e.level || 'INFO';
    const msg = escHtml(e.msg || '');
    return `<div class="log-line"><span class="log-ts">${ts}</span><span class="log-src ${e._source}">${e._source}</span><span class="log-level ${lvl}">${lvl}</span><span class="log-msg">${msg}</span></div>`;
  }).join('');
  document.getElementById('log-count').textContent = logLineCount + ' lines';
  if (document.getElementById('autoscroll').checked) feed.scrollTop = feed.scrollHeight;
}

// Backfill on load
fetch('/api/logs/backfill').then(r=>r.json()).then(lines => {
  lines.forEach(addLogLine);
  renderLogFeed();
});

// SSE stream
const sse = new EventSource('/api/logs/stream');
sse.onmessage = e => {
  try {
    const entry = JSON.parse(e.data);
    addLogLine(entry);
    // auto-render only if activity panel is active
    if (document.getElementById('panel-activity').classList.contains('active')) {
      renderLogFeed();
    }
  } catch {}
};

// ── Signals ────────────────────────────────────────────────────────────────
function setSigFilter(f, el) {
  sigFilter = f;
  document.querySelectorAll('.sig-filter').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  renderSignals();
}
function renderSignals() {
  const grid = document.getElementById('signals-grid');
  let sigs = allSignals;
  if (sigFilter !== 'ALL') {
    sigs = sigs.filter(s => {
      const sig = (s.signal || '').toUpperCase();
      if (sigFilter === 'BUY') return /BUY|CONFIRM|STRONG/.test(sig);
      if (sigFilter === 'HOLD') return /HOLD|WATCH/.test(sig);
      if (sigFilter === 'SKIP') return /SKIP|BREAK/.test(sig);
      return true;
    });
  }
  if (!sigs.length) { grid.innerHTML = '<div class="signals-empty">No signals match this filter.</div>'; return; }
  grid.innerHTML = sigs.map(s => {
    const sig = (s.signal || '').toUpperCase().replace(/\s+/g, '_');
    const sigClass = /BUY|CONFIRM/.test(sig) ? 'buy' : /HOLD|WATCH/.test(sig) ? 'hold' : 'skip';
    const factors = [
      { label: 'T', val: s.trend, max: 2 },
      { label: 'M', val: s.momentum, max: 2 },
      { label: 'V', val: s.volatility, max: 1 },
      { label: 'L', val: s.liquidity, max: 2 },
      { label: 'F', val: s.fundamental, max: 2 },
    ];
    const bars = factors.map(f => {
      if (f.val == null) return '';
      const pct = Math.min(100, Math.max(0, (f.val / f.max * 50 + 50)));
      const color = f.val >= 0 ? 'var(--green)' : 'var(--red)';
      return `<div class="sig-bar-wrap"><div class="sig-bar-label">${f.label}</div><div class="sig-bar" style="background:var(--bg-3)"><div style="position:absolute;bottom:0;left:0;right:0;height:${pct}%;background:${color};opacity:.7;border-radius:2px"></div><span style="position:relative;z-index:1;font-size:8px">${f.val!=null?f.val.toFixed(1):''}</span></div></div>`;
    }).join('');
    return `<div class="sig-card ${sig}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div class="sig-ticker">${s.ticker}</div>
        <div class="sig-signal ${sigClass}">${s.signal || '—'}</div>
      </div>
      ${bars ? `<div class="sig-bars">${bars}</div>` : ''}
      ${s.total != null ? `<div class="sig-conf">total <span>${s.total >= 0 ? '+' : ''}${s.total.toFixed(2)}</span></div>` : ''}
      ${s.confidence != null ? `<div class="sig-conf">conf <span>${(s.confidence*100).toFixed(0)}%</span></div>` : ''}
    </div>`;
  }).join('');
}
async function loadSignals() {
  const res = await fetch('/api/signals');
  const data = await res.json();
  allSignals = data.signals;
  document.getElementById('sig-date').textContent = data.date;
  renderSignals();
}

// ── Options ────────────────────────────────────────────────────────────────
async function loadOptions() {
  const tbody = document.getElementById('opt-tbody');
  tbody.innerHTML = '<tr><td colspan="8" class="opt-empty">Loading…</td></tr>';
  try {
    const res = await fetch('/api/options');
    const data = await res.json();
    if (data.error) {
      tbody.innerHTML = `<tr><td colspan="8" class="opt-empty">Could not load options: ${escHtml(data.error)}</td></tr>`;
      return;
    }
    renderOptions(data);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="opt-empty">Error: ${escHtml(String(e))}</td></tr>`;
  }
}
function renderOptions(data) {
  const s = data.summary || {count:0, total_value:0, total_pl:0};
  document.getElementById('opt-count').textContent = s.count;
  document.getElementById('opt-value').textContent = '$' + Number(s.total_value).toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2});
  const plEl = document.getElementById('opt-pl');
  plEl.textContent = (s.total_pl >= 0 ? '+$' : '-$') + Math.abs(s.total_pl).toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2});
  plEl.className = 'opt-stat-val ' + (s.total_pl >= 0 ? 'pos' : 'neg');

  const tbody = document.getElementById('opt-tbody');
  const rows = data.positions || [];
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="opt-empty">No open options positions.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(p => {
    const plClass = p.unrealized_pl >= 0 ? 'pos' : 'neg';
    const plStr = (p.unrealized_pl >= 0 ? '+$' : '-$') + Math.abs(p.unrealized_pl).toFixed(2);
    const pctStr = (p.pl_pct >= 0 ? '+' : '') + p.pl_pct.toFixed(1) + '%';
    const strike = p.strike != null ? '$' + Number(p.strike).toFixed(2) : '—';
    return `<tr>
      <td><div class="under">${escHtml(p.underlying || '')}</div><div class="occ">${escHtml(p.symbol)}</div></td>
      <td>${p.type ? `<span class="opt-type ${p.type}">${p.type}</span>` : '—'}</td>
      <td>${strike}</td>
      <td>${escHtml(p.expiry || '—')}</td>
      <td class="num">${p.qty}</td>
      <td class="num">$${Number(p.market_value).toFixed(2)}</td>
      <td class="num opt-pl ${plClass}">${plStr}</td>
      <td class="num opt-pl ${plClass}">${pctStr}</td>
    </tr>`;
  }).join('');
}

// ── Watchlist ─────────────────────────────────────────────────────────────
async function loadWatchlist() {
  const res = await fetch('/api/watchlist');
  const data = await res.json();
  renderWatchlist(data.tickers);
}
function renderWatchlist(tickers) {
  document.getElementById('wl-count').textContent = `(${tickers.length} tickers)`;
  document.getElementById('ticker-grid').innerHTML = tickers.map(t => `
    <div class="ticker-chip" id="chip-${t}">
      ${t}
      <button onclick="removeTicker('${t}')" title="Remove">×</button>
    </div>`).join('') || '<span style="color:var(--muted)">No tickers</span>';
}
async function addTicker() {
  const inp = document.getElementById('ticker-input');
  const ticker = inp.value.trim().toUpperCase();
  if (!ticker) return;
  const res = await fetch('/api/watchlist/add', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticker}) });
  const data = await res.json();
  if (data.tickers) { renderWatchlist(data.tickers); inp.value = ''; }
}
async function removeTicker(ticker) {
  const chip = document.getElementById('chip-' + ticker);
  if (chip) chip.style.opacity = '.4';
  const res = await fetch('/api/watchlist/remove', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticker}) });
  const data = await res.json();
  if (data.tickers) renderWatchlist(data.tickers);
}

// ── Settings ──────────────────────────────────────────────────────────────
async function loadSettings() {
  const res = await fetch('/api/settings');
  const data = await res.json();
  const env = data.env;
  const pack = data.pack;
  const safeKeys = ['ENV','CLAUDE_MODEL','WEBHOOK_PORT','MAX_POSITION_SIZE_USD','DAILY_LOSS_LIMIT_USD','ALPACA_BASE_URL'];
  const body = document.getElementById('settings-body');
  body.innerHTML = `
    <div class="settings-grid">
      ${safeKeys.map(k => `
        <div class="setting-row">
          <label>${k}</label>
          <input id="env-${k}" value="${escHtml(env[k] || '')}" placeholder="(not set)">
        </div>`).join('')}
    </div>
    <div class="settings-save" style="margin-top:12px">
      <button class="btn-primary" onclick="saveSettings()">Save Changes</button>
      <span class="save-msg" id="save-msg">✓ Saved</span>
    </div>`;

  const packEl = document.getElementById('pack-info');
  if (pack && pack.pack) {
    const regimeMap = pack.regime_map || {};
    const rows = Object.entries(regimeMap).map(([regime, cfg]) =>
      `<div class="pack-key">${regime}</div><div class="pack-val">→ ${cfg.template} / ${cfg.version}</div>`
    ).join('');
    packEl.innerHTML = `
      <div class="pack-key">Pack</div><div class="pack-val">${pack.pack}</div>
      <div class="pack-key">Version</div><div class="pack-val">${pack.version}</div>
      <div class="pack-key">Default</div><div class="pack-val">${pack.default_template} / ${pack.default_version}</div>
      ${rows}`;
  } else {
    packEl.innerHTML = '<div style="color:var(--muted)">pack.json not found</div>';
  }
}
async function saveSettings() {
  const safeKeys = ['ENV','CLAUDE_MODEL','WEBHOOK_PORT','MAX_POSITION_SIZE_USD','DAILY_LOSS_LIMIT_USD','ALPACA_BASE_URL'];
  const payload = {};
  safeKeys.forEach(k => {
    const el = document.getElementById('env-' + k);
    if (el && el.value.trim()) payload[k] = el.value.trim();
  });
  const res = await fetch('/api/settings', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const msg = document.getElementById('save-msg');
  if (res.ok) {
    msg.classList.add('show');
    setTimeout(() => msg.classList.remove('show'), 2500);
  } else {
    msg.textContent = '✗ Error';
    msg.style.color = 'var(--red)';
    msg.classList.add('show');
    setTimeout(() => { msg.classList.remove('show'); msg.textContent = '✓ Saved'; msg.style.color = ''; }, 3000);
  }
}


// ==================== Portfolio panel (OpenAlice-ported) ====================
let equityChart = null;
let currentRange = '24H';
let lastRefresh = null;

function fmtUsd(n)  { return '$' + Number(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function fmtPnl(n)  { const s = n >= 0 ? '+' : '-'; return s + '$' + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function fmtPct(n)  { return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'; }

function setRange(rng, btn) {
  currentRange = rng;
  document.querySelectorAll('.range-pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadEquityCurve(rng);
}

async function loadPortfolio() {
  try {
    const [acct, pos] = await Promise.all([
      fetch('/api/account').then(r => r.json()),
      fetch('/api/positions').then(r => r.json()),
    ]);
    renderHero(acct);
    renderPositions(pos.positions || []);
    renderMovers(pos.positions || []);
    renderMix(acct, pos.positions || []);
    renderRisk(acct, pos.positions || []);
    lastRefresh = new Date();
    updateLastUpdated();
  } catch (e) { console.error('loadPortfolio:', e); }
  loadEquityCurve(currentRange);
}

function renderHero(a) {
  if (a.error) {
    document.getElementById('hero-equity').textContent = 'Error';
    document.getElementById('hero-delta').textContent = a.error;
    return;
  }
  document.getElementById('hero-equity').textContent = fmtUsd(a.equity);
  document.getElementById('hero-cash').textContent   = fmtUsd(a.cash);
  document.getElementById('hero-bp').textContent     = fmtUsd(a.buying_power);
  const upl = a.unrealized_pl;
  const uplEl = document.getElementById('hero-upl');
  uplEl.textContent = fmtPnl(upl);
  uplEl.className = 'metric-value sm ' + (upl > 0 ? 'up' : upl < 0 ? 'down' : '');
  const deltaEl = document.getElementById('hero-delta');
  const d = a.today_pl;
  if (d === 0) {
    deltaEl.textContent = '— today';
    deltaEl.className = 'metric-delta';
  } else {
    const arrow = d >= 0 ? '▲' : '▼';
    deltaEl.textContent = arrow + ' ' + fmtPnl(d) + ' (' + fmtPct(a.today_pl_pct) + ') today';
    deltaEl.className = 'metric-delta ' + (d > 0 ? 'up' : 'down');
  }
}

function renderPositions(positions) {
  const tbody = document.getElementById('positions-tbody');
  if (!positions.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="pos-empty">No open positions.</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    const pl = Number(p.unrealized_pl);
    const pct = Number(p.unrealized_plpc);
    const plClass = pl >= 0 ? 'pos' : 'neg';
    const tagClass = p.side === 'short' ? 'short' : '';
    const tagTxt = p.side === 'short' ? 'SHORT' : p.tag;
    return '<tr>' +
      '<td><span class="sym-name">' + escHtml(p.symbol) + '</span>' +
        '<span class="sym-tag ' + tagClass + '">' + tagTxt + '</span></td>' +
      '<td class="num">' + Number(p.qty).toLocaleString(undefined, {maximumFractionDigits: 6}) + '</td>' +
      '<td class="num">' + fmtUsd(p.avg_cost) + '</td>' +
      '<td class="num">' + fmtUsd(p.market_price) + '</td>' +
      '<td class="num">' + fmtUsd(p.market_value) + '</td>' +
      '<td class="num pos-pl ' + plClass + '">' + fmtPnl(pl) + '</td>' +
      '<td class="num pos-pl ' + plClass + '">' + fmtPct(pct) + '</td>' +
    '</tr>';
  }).join('');
}

function fmtXTick(ms, range) {
  const d = new Date(ms);
  if (range === '1H' || range === '6H' || range === '24H') {
    return d.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});
  } else if (range === '7D') {
    return d.toLocaleDateString(undefined, {weekday: 'short', day: 'numeric'});
  } else {
    return d.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
  }
}

async function loadEquityCurve(range) {
  const empty = document.getElementById('equity-chart-empty');
  try {
    const data = await fetch('/api/equity_curve?range=' + range).then(r => r.json());
    const points = data.points || [];
    if (!points.length) {
      empty.style.display = 'flex';
      empty.textContent = 'No equity history for this range.';
      if (equityChart) { equityChart.destroy(); equityChart = null; }
      return;
    }
    empty.style.display = 'none';
    const chartData = points.map(p => ({x: p.t, y: p.equity}));
    if (equityChart) {
      equityChart.data.datasets[0].data = chartData;
      equityChart.options.scales.x.ticks.callback = (v) => fmtXTick(v, range);
      equityChart.update('none');
      return;
    }
    const canvas = document.getElementById('equity-chart');
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 240);
    gradient.addColorStop(0, 'rgba(88, 166, 255, 0.35)');
    gradient.addColorStop(1, 'rgba(88, 166, 255, 0)');
    equityChart = new Chart(ctx, {
      type: 'line',
      data: { datasets: [{
        data: chartData,
        borderColor: '#58a6ff',
        backgroundColor: gradient,
        borderWidth: 1.5,
        fill: true,
        tension: 0.25,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#58a6ff',
        pointHoverBorderColor: '#161b22',
        pointHoverBorderWidth: 2,
      }]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#161b22',
            borderColor: '#30363d',
            borderWidth: 1,
            titleColor: '#8b949e',
            bodyColor: '#e6edf3',
            padding: 10,
            displayColors: false,
            callbacks: {
              title: (items) => new Date(items[0].parsed.x).toLocaleString(),
              label:  (item)  => '$' + Number(item.parsed.y).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}),
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            grid: { display: false },
            border: { color: '#30363d' },
            ticks: {
              color: '#8b949e',
              font: { size: 10 },
              maxTicksLimit: 6,
              callback: (v) => fmtXTick(v, range),
            },
          },
          y: {
            position: 'right',
            grid: { color: 'rgba(48, 54, 61, 0.4)' },
            border: { display: false },
            ticks: {
              color: '#8b949e',
              font: { size: 10 },
              maxTicksLimit: 5,
              callback: (v) => '$' + Number(v).toLocaleString(undefined, {maximumFractionDigits: 0}),
            },
          },
        },
      },
    });
  } catch (e) {
    console.error('loadEquityCurve:', e);
    empty.style.display = 'flex';
    empty.textContent = 'Could not load curve.';
  }
}

function updateLastUpdated() {
  const el = document.getElementById('last-updated');
  if (!el || !lastRefresh) return;
  const secs = Math.floor((Date.now() - lastRefresh.getTime()) / 1000);
  let txt;
  if (secs < 5)       txt = 'just now';
  else if (secs < 60) txt = secs + 's ago';
  else if (secs < 3600) txt = Math.floor(secs / 60) + 'm ago';
  else                txt = Math.floor(secs / 3600) + 'h ago';
  el.textContent = 'updated ' + txt;
}
setInterval(updateLastUpdated, 1000);


// ==================== Portfolio sub-widgets ====================

function renderMovers(positions) {
  // Top 3 by gain pct, top 3 by loss pct
  const withPct = positions.map(p => ({
    sym: p.symbol,
    tag: p.tag,
    pl: Number(p.unrealized_pl),
    pct: Number(p.unrealized_plpc),
  }));
  const gainers = [...withPct].filter(p => p.pct > 0).sort((a,b) => b.pct - a.pct).slice(0, 3);
  const losers  = [...withPct].filter(p => p.pct < 0).sort((a,b) => a.pct - b.pct).slice(0, 3);
  
  const row = (p, cls) =>
    '<div class="mover-row">' +
      '<span class="mover-sym">' + escHtml(p.sym) + '</span>' +
      '<span class="mover-pct ' + cls + '">' + fmtPct(p.pct) + '</span>' +
    '</div>';
  
  document.getElementById('movers-up').innerHTML =
    gainers.length ? gainers.map(p => row(p, 'pos')).join('')
                   : '<div class="mover-empty">No gainers today.</div>';
  document.getElementById('movers-down').innerHTML =
    losers.length ? losers.map(p => row(p, 'neg')).join('')
                  : '<div class="mover-empty">No losers today.</div>';
}

function renderMix(account, positions) {
  // Asset class breakdown by market value + cash
  const equity = Number(account.equity) || 0;
  const cash = Number(account.cash) || 0;
  let stkVal = 0, cryptoVal = 0;
  for (const p of positions) {
    const v = Number(p.market_value) || 0;
    if (p.tag === 'CRYPTO') cryptoVal += v;
    else stkVal += v;
  }
  // Positive only for bar
  const total = Math.max(stkVal + cryptoVal + cash, 1);
  const stkPct    = (stkVal / total) * 100;
  const cryptoPct = (cryptoVal / total) * 100;
  const cashPct   = (cash / total) * 100;
  
  const rows = document.getElementById('mix-rows');
  const parts = [];
  if (stkVal > 0)    parts.push({label: 'Equities', val: stkVal, pct: stkPct});
  if (cryptoVal > 0) parts.push({label: 'Crypto',   val: cryptoVal, pct: cryptoPct});
  parts.push({label: 'Cash', val: cash, pct: cashPct});
  
  rows.innerHTML = parts.map(p =>
    '<div class="mix-row">' +
      '<span class="mix-label">' + p.label + '</span>' +
      '<span class="mix-val">' + fmtUsd(p.val) + ' <span style="color:var(--text-muted);font-weight:400;">· ' + p.pct.toFixed(1) + '%</span></span>' +
    '</div>'
  ).join('');
  
  const bar = document.getElementById('mix-bar');
  bar.innerHTML =
    (stkVal    > 0 ? '<div class="mix-bar-seg stk"    style="width:' + stkPct    + '%"></div>' : '') +
    (cryptoVal > 0 ? '<div class="mix-bar-seg crypto" style="width:' + cryptoPct + '%"></div>' : '') +
    (cash > 0      ? '<div class="mix-bar-seg cash"   style="width:' + cashPct   + '%"></div>' : '');
}

function renderRisk(account, positions) {
  const equity = Number(account.equity) || 0;
  const cash   = Number(account.cash) || 0;
  
  // Largest single position (by abs market value)
  let largestSym = '—', largestPct = 0;
  if (positions.length && equity > 0) {
    const top = [...positions].sort((a, b) =>
      Math.abs(Number(b.market_value)) - Math.abs(Number(a.market_value))
    )[0];
    largestSym = top.symbol;
    largestPct = (Math.abs(Number(top.market_value)) / equity) * 100;
  }
  
  const cashPct     = equity > 0 ? (cash / equity) * 100 : 0;
  const deployedPct = 100 - cashPct;
  
  document.getElementById('risk-rows').innerHTML = [
    {label: 'Open positions', val: positions.length},
    {label: 'Deployed',       val: deployedPct.toFixed(1) + '%'},
    {label: 'Largest',        val: escHtml(largestSym) + ' · ' + largestPct.toFixed(1) + '%'},
  ].map(r =>
    '<div class="risk-row">' +
      '<span class="risk-label">' + r.label + '</span>' +
      '<span class="risk-val">' + r.val + '</span>' +
    '</div>'
  ).join('');
}

// ── Refresh all ────────────────────────────────────────────────────────────
function refreshAll() {
  if (document.getElementById('panel-portfolio').classList.contains('active')) loadPortfolio();
  loadStatus();
  loadSchedule();
  loadGlance();
  if (document.getElementById('panel-signals').classList.contains('active')) loadSignals();
  if (document.getElementById('panel-options').classList.contains('active')) loadOptions();
  if (document.getElementById('panel-watchlist').classList.contains('active')) loadWatchlist();
  if (document.getElementById('panel-settings').classList.contains('active')) loadSettings();
}

// ── Utilities ─────────────────────────────────────────────────────────────
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ── Init ──────────────────────────────────────────────────────────────────
loadPortfolio();
setInterval(loadPortfolio, 30000);
loadStatus();
loadSchedule();
loadGlance();
setInterval(loadStatus, 30000);
setInterval(loadSchedule, 60000);
</script>
</body>
</html>
"""


@app.route("/health")
def health():
    return Response("ready", status=200, mimetype="text/plain")


@app.route("/")
def index():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start log tailer background thread
    t = threading.Thread(target=_log_tailer, daemon=True)
    t.start()

    port = int(os.getenv("DASHBOARD_PORT", "5001"))
    print(f"Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False,
            threaded=True)
