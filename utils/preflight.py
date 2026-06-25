"""
utils/preflight.py -- Startup dependency diagnostics.

validate_config() only checks that credentials EXIST. This module checks they
actually WORK by making a real, cheap call to each critical dependency before
the scheduler loop starts -- so a bad key / unreachable broker fails loudly at
boot instead of silently failing mid-scan an hour later.

Severity policy:
  * Alpaca account      -> CRITICAL. No broker = can't trade or manage exits.
                           Fatal by default (set die=True). A transient blip is
                           handled at runtime by fail-closed logic; this is for
                           a genuinely broken setup (wrong key, wrong env).
  * Market data         -> CRITICAL but degradable. Has a cache fallback at
                           runtime, so we warn hard rather than always kill.
  * Anthropic / Claude  -> WARNING only. Claude scoring is ONE input; rule-based
                           strategies (chart/VWAP/crypto) run without it.

Each check has a short timeout so a hang can't stall startup forever.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from loguru import logger


@dataclass
class CheckResult:
    name:     str
    ok:       bool
    critical: bool
    detail:   str
    latency_ms: float


def _timed(fn):
    t0 = time.time()
    try:
        detail = fn()
        return True, detail, (time.time() - t0) * 1000
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", (time.time() - t0) * 1000


def check_alpaca() -> CheckResult:
    """Verify the Alpaca account is reachable AND authenticated."""
    def _do():
        from brokers.alpaca import get_account
        acct = get_account()
        equity = acct.get("equity", "?")
        bp = acct.get("buying_power", "?")
        return f"account reachable (equity={equity}, buying_power={bp})"
    ok, detail, ms = _timed(_do)
    return CheckResult("Alpaca account", ok, True, detail, ms)


def check_market_data() -> CheckResult:
    """Verify market data is flowing (one cheap OHLCV pull)."""
    def _do():
        from data.market_data import get_ohlcv
        df = get_ohlcv("SPY", period="5d", interval="1d")
        if df is None or df.empty:
            raise RuntimeError("empty OHLCV for SPY")
        return f"OHLCV ok ({len(df)} SPY bars, last close={float(df['close'].iloc[-1]):.2f})"
    ok, detail, ms = _timed(_do)
    return CheckResult("Market data", ok, True, detail, ms)


def check_anthropic() -> CheckResult:
    """Verify the Anthropic API key works with a tiny completion. Non-fatal."""
    def _do():
        from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        return f"Claude reachable (model={CLAUDE_MODEL}, id={getattr(resp,'id','?')[:18]})"
    ok, detail, ms = _timed(_do)
    return CheckResult("Anthropic / Claude", ok, False, detail, ms)


def run_preflight(die: bool = True, check_claude: bool = True) -> list[CheckResult]:
    """
    Run all startup diagnostics. Logs a clear PASS/FAIL line per check.

    If die=True and any CRITICAL check fails, raises SystemExit so the process
    stops before the scheduler loop (and the watchdog can surface it). Non-
    critical failures only warn. Returns the list of results either way.
    """
    logger.info("=== PREFLIGHT: verifying broker + data + AI connectivity ===")
    checks = [check_alpaca, check_market_data]
    if check_claude:
        checks.append(check_anthropic)

    results: list[CheckResult] = []
    critical_failed: list[CheckResult] = []
    for chk in checks:
        r = chk()
        results.append(r)
        tag = "PASS" if r.ok else ("FAIL-CRIT" if r.critical else "WARN")
        line = f"PREFLIGHT [{tag}] {r.name}: {r.detail} ({r.latency_ms:.0f}ms)"
        if r.ok:
            logger.info(line)
        elif r.critical:
            logger.critical(line)
            critical_failed.append(r)
        else:
            logger.warning(line)

    if critical_failed:
        names = ", ".join(r.name for r in critical_failed)
        _alert(f"Preflight FAILED: {names} unreachable at startup. Bot did not start.")
        if die:
            logger.critical(
                f"=== PREFLIGHT FAILED ({names}) -- refusing to start. "
                f"Fix the dependency and restart. ==="
            )
            raise SystemExit(2)
        logger.error(f"=== PREFLIGHT had critical failures ({names}) but die=False -- continuing. ===")
    else:
        logger.info("=== PREFLIGHT PASSED -- all critical dependencies reachable ===")
    return results


def _alert(message: str) -> None:
    """Best-effort Telegram alert (startup failures should reach the operator)."""
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            import requests as _req
            _req.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": f"\u26a0\ufe0f {message}",
                      "parse_mode": "HTML"},
                timeout=5,
            )
    except Exception:
        pass
