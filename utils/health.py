"""
utils/health.py -- Lightweight health/readiness + metrics HTTP server.

Runs in a background daemon thread so it does not interfere with the
main process.  Exposes three endpoints:

  GET /health   -- always 200 {"status":"ok"} while the process is alive
  GET /ready    -- 200 {"status":"ready"} once mark_ready() has been called,
                   503 {"status":"starting"} before that
  GET /metrics  -- Prometheus text exposition (from utils.metrics)

Usage:
    from utils.health import HealthServer
    health = HealthServer(port=9000, name="bot")
    health.start()
    # ... startup work ...
    health.mark_ready()     # /ready now returns 200
"""
from __future__ import annotations
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from loguru import logger


class HealthServer:
    def __init__(self, port: int = 9000, name: str = "app") -> None:
        self._port    = port
        self._name    = name
        self._ready   = False
        self._started = datetime.now(timezone.utc).isoformat()
        self._server: HTTPServer | None = None

    def mark_ready(self) -> None:
        self._ready = True
        logger.debug(f"Health: {self._name} marked ready on :{self._port}")

    def start(self) -> None:
        """Start the health server in a background daemon thread."""
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                content_type = "application/json"
                if self.path in ("/health", "/"):
                    body = json.dumps({
                        "status":   "ok",
                        "name":     outer._name,
                        "started":  outer._started,
                        "uptime_s": round(
                            (datetime.now(timezone.utc) -
                             datetime.fromisoformat(outer._started)).total_seconds()
                        ),
                    }).encode()
                    self.send_response(200)
                elif self.path == "/ready":
                    if outer._ready:
                        body = json.dumps({"status": "ready"}).encode()
                        self.send_response(200)
                    else:
                        body = json.dumps({"status": "starting"}).encode()
                        self.send_response(503)
                elif self.path == "/metrics":
                    try:
                        from utils.metrics import render_text
                        body = render_text().encode()
                        content_type = "text/plain; version=0.0.4; charset=utf-8"
                        self.send_response(200)
                    except Exception as exc:
                        body = f"# metrics error: {exc}\n".encode()
                        content_type = "text/plain"
                        self.send_response(500)
                else:
                    body = b'{"error":"not found"}'
                    self.send_response(404)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):  # silence access log
                pass

        def _serve():
            for attempt in range(5):
                try:
                    port = self._port + attempt

                    # allow_reuse_address lets the new process reclaim the port
                    # immediately after a restart instead of waiting for TIME_WAIT
                    # to expire (up to 4 min on Windows).  Without this, the
                    # fallback binds to port+1/+2/etc. and the watchdog's fixed
                    # health-check URL never gets a response → crash-loop.
                    class _ReuseHTTPServer(HTTPServer):
                        allow_reuse_address = True

                    srv = _ReuseHTTPServer(("0.0.0.0", port), _Handler)
                    self._server = srv
                    if attempt:
                        logger.warning(
                            f"Health: port {self._port} busy, using {port} instead"
                        )
                    logger.debug(f"Health server listening on :{port} ({self._name})")
                    srv.serve_forever()
                    return
                except OSError:
                    time.sleep(0.2)
            logger.warning(
                f"Health: could not bind to port {self._port}-{self._port + 4}, skipping."
            )

        t = threading.Thread(target=_serve, daemon=True, name=f"health-{self._name}")
        t.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
