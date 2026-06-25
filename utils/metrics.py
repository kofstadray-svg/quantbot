"""
utils/metrics.py — Lightweight, zero-dependency Prometheus-compatible metrics.

Exposes counters, gauges, and histograms in the standard Prometheus text
exposition format so any scraper (Prometheus, Grafana Agent, VictoriaMetrics,
or just `curl`) can consume them.

The /metrics HTTP path is automatically added to HealthServer — no extra
wiring needed.  Just call the module-level helpers anywhere in the codebase:

    from utils.metrics import counter, gauge, histogram

    counter("trades_placed_total", {"ticker": "AAPL", "side": "BUY"})
    gauge("daily_pnl_usd", {}, -23.50)

    with histogram("scan_duration_seconds", {"scan": "nasdaq"}):
        run_scan()

All operations are thread-safe.  The registry is a module-level singleton
so any import gets the same state.

No external packages required — uses only stdlib.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator


# ── Internal registry ──────────────────────────────────────────────────────────

class _Registry:
    """Thread-safe in-memory metric store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {metric_name: {"help": str, "type": str, "series": {label_key: value}}}
        self._metrics: dict[str, dict] = {}

    def _label_key(self, labels: dict[str, str]) -> str:
        """Stable string key for a label set."""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def _label_str(self, name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{pairs}}}"

    def _ensure(self, name: str, metric_type: str, help_text: str) -> None:
        if name not in self._metrics:
            self._metrics[name] = {
                "help":   help_text,
                "type":   metric_type,
                "series": defaultdict(float),
            }

    def inc(self, name: str, labels: dict[str, str], amount: float, help_text: str) -> None:
        with self._lock:
            self._ensure(name, "counter", help_text)
            self._metrics[name]["series"][self._label_key(labels)] += amount

    def set(self, name: str, labels: dict[str, str], value: float, help_text: str) -> None:
        with self._lock:
            self._ensure(name, "gauge", help_text)
            self._metrics[name]["series"][self._label_key(labels)] = value

    def observe(self, name: str, labels: dict[str, str], value: float, help_text: str) -> None:
        """Record a histogram sample (stored as _sum / _count / simple buckets)."""
        with self._lock:
            base = f"{name}"
            self._ensure(f"{base}_sum",   "gauge", f"{help_text} (sum)")
            self._ensure(f"{base}_count", "counter", f"{help_text} (count)")
            lk = self._label_key(labels)
            self._metrics[f"{base}_sum"]["series"][lk]   += value
            self._metrics[f"{base}_count"]["series"][lk] += 1

    def render(self) -> str:
        """Return full Prometheus text exposition."""
        lines: list[str] = []
        with self._lock:
            snapshot = {
                n: {
                    "help":   m["help"],
                    "type":   m["type"],
                    "series": dict(m["series"]),
                }
                for n, m in self._metrics.items()
            }
        for name, meta in sorted(snapshot.items()):
            lines.append(f"# HELP {name} {meta['help']}")
            lines.append(f"# TYPE {name} {meta['type']}")
            for label_key, value in sorted(meta["series"].items()):
                if label_key:
                    # Rebuild label string from stored key
                    lines.append(f"{name}{{{label_key}}} {_fmt(value)}")
                else:
                    lines.append(f"{name} {_fmt(value)}")
        return "\n".join(lines) + "\n"


def _fmt(v: float) -> str:
    """Format a float for Prometheus — integer values without decimal."""
    if v == int(v):
        return str(int(v))
    return f"{v:.6g}"


_registry = _Registry()


# ── Public API ─────────────────────────────────────────────────────────────────

def counter(
    name: str,
    labels: dict[str, str] | None = None,
    inc: float = 1,
    help: str = "",
) -> None:
    """Increment a counter by inc (default 1)."""
    _registry.inc(name, labels or {}, inc, help or name)


def gauge(
    name: str,
    labels: dict[str, str] | None = None,
    value: float = 0,
    help: str = "",
) -> None:
    """Set a gauge to value."""
    _registry.set(name, labels or {}, value, help or name)


def histogram_observe(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
    help: str = "",
) -> None:
    """Record a single histogram observation."""
    _registry.observe(name, labels or {}, value, help or name)


@contextmanager
def histogram(
    name: str,
    labels: dict[str, str] | None = None,
    help: str = "",
) -> Generator[None, None, None]:
    """
    Context manager that times the block and records it as a histogram.

        with histogram("scan_duration_seconds", {"scan": "nasdaq"}):
            run_scan()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _registry.observe(name, labels or {}, elapsed, help or name)


def render_text() -> str:
    """Return full Prometheus text exposition string."""
    return _registry.render()


# ── Pre-declared bot metrics ───────────────────────────────────────────────────
# Calling these at import time registers the metric so it appears in /metrics
# even before the first event fires (value = 0).

def _init_defaults() -> None:
    gauge("bot_up", {}, 0,
          help="1 if bot process is running")
    counter("trades_placed_total", {"side": "BUY", "market": "stock"},
            inc=0, help="Total orders submitted to Alpaca")
    counter("trades_placed_total", {"side": "BUY", "market": "crypto"},
            inc=0, help="Total orders submitted to Alpaca")
    counter("trades_placed_total", {"side": "SELL", "market": "stock"},
            inc=0, help="Total orders submitted to Alpaca")
    counter("scans_run_total", {"scan": "init"},
            inc=0, help="Total scan runs by scan name")
    counter("scan_errors_total", {"scan": "init"},
            inc=0, help="Total errors during scans")
    gauge("daily_pnl_usd", {}, 0,
          help="Running realized P&L for the current trading day (USD)")
    gauge("open_positions", {}, 0,
          help="Number of currently open positions")

_init_defaults()
