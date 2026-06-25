"""
morning_scan.py — Run a one-shot morning scan: Unified screener + Chart Patterns.

HOW TO RUN:
  python morning_scan.py

Runs each scan immediately and exits. Does NOT start the scheduler.
Safe to run at any time — the broker will skip symbols already held.
"""
from loguru import logger
from utils.risk import reset_daily_loss

from utils.logging import setup_logging
from utils.health import HealthServer
setup_logging("morning_scan")
_health = HealthServer(port=9101, name="morning-scan")
_health.start()


def run_unified() -> None:
    """
    Two-layer scan: Momentum screener pre-filters, then Claude AI confirms.
    STRONG BUY  -> buy stock + buy call (28-50 DTE, ~2% OTM)
    BREAKDOWN   -> buy put  (28-50 DTE, ~2% OTM)
    """
    from agents.unified_screener import screen as unified_screen, summary
    from brokers.alpaca import market_buy
    from brokers.alpaca_options import buy_call, buy_put
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    logger.info(f"Unified scan: {len(watchlist)} tickers")

    results = unified_screen(watchlist)
    summary(results)

    def _journal_indicators(r):
        try:
            from agents.stock_screener import _compute_indicators
            full = _compute_indicators(r["ticker"])
            if full:
                return full
        except Exception:
            pass
        return {
            "rsi_14":      r.get("rsi"),
            "rel_volume":  r.get("rel_volume"),
            "obv_trend":   None,
            "macd_signal": None,
            "golden_cross": None,
            "stoch_k":     None,
            "volume_rsi":  None,
            "bb_position": None,
        }

    for r in results:
        if r["unified_signal"] == "STRONG BUY":
            entry = ", ".join(r["entry_timing"])
            logger.info(
                "AUTO-TRADE -> {}  [Mom={} | AI={} score={}]  Entry={}  -- {}".format(
                    r['ticker'], r['mom_signal'], r['ai_signal'],
                    r['ai_score'], entry, r['ai_reason']
                )
            )
            order_result = market_buy(r["ticker"], 100)
            try:
                from utils.trade_journal import record_signal
                entry_price = r.get("price") or r.get("close") or None
                order_id    = (order_result or {}).get("id")
                record_signal(
                    r["ticker"],
                    _journal_indicators(r),
                    r.get("ai_score") or 0,
                    entry_price=entry_price,
                    alpaca_order_id=order_id,
                )
            except Exception as _je:
                logger.warning(f"Journal record failed: {_je}")
            opt = buy_call(r["ticker"], notional_usd=50)
            if opt:
                logger.info(
                    "OPTIONS CALL -> {}  {}  strike={}  exp={}".format(
                        r['ticker'], opt['symbol'], opt['strike'], opt['expiry']
                    )
                )
            else:
                logger.warning(f"OPTIONS CALL -> {r['ticker']} -- no contract found, skipping.")

        elif r["unified_signal"] == "BREAKDOWN":
            logger.info(
                "BREAKDOWN -> {}  RSI={}  Gap={:+}%  RelVol={}x -- buying put".format(
                    r['ticker'], r.get('rsi','?'), r.get('gap_pct','?'), r.get('rel_volume','?')
                )
            )
            opt = buy_put(r["ticker"], notional_usd=50)
            if opt:
                logger.info(
                    "OPTIONS PUT  -> {}  {}  strike={}  exp={}".format(
                        r['ticker'], opt['symbol'], opt['strike'], opt['expiry']
                    )
                )
            else:
                logger.warning(f"OPTIONS PUT  -> {r['ticker']} -- no contract found, skipping.")

        elif r["unified_signal"] == "CONFIRMED":
            logger.info(
                f"HOLD -> {r['ticker']}  both screeners agree but no entry setup yet -- watching."
            )


def run_chart_patterns() -> None:
    from agents.chart_pattern_screener import screen_chart_patterns
    from brokers.alpaca import market_buy_with_stop
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist
    from data.custom_watchlist import CUSTOM_WATCHLIST
    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    logger.info(f"Chart pattern scan: {len(watchlist)} tickers")
    hits = screen_chart_patterns(watchlist)
    if not hits:
        logger.info("Chart patterns: no confirmed breakouts today.")
        return
    for h in hits:
        logger.info(
            "Chart  | {:6s} [{:9s}] {:25s} -> {}".format(
                h['ticker'], h['state'], h['pattern'], h['action']
            )
        )
        if h["state"] == "confirmed" and h["action"] == "BUY":
            logger.info(f"AUTO-TRADE -> {h['ticker']} ({h['pattern']})")
            market_buy_with_stop(h["ticker"], 100, stop_pct=0.10, take_profit_pct=0.12)


if __name__ == "__main__":
    print("\n========================================")
    print("  MORNING SCAN")
    print("========================================\n")

    reset_daily_loss()
    _health.mark_ready()
    logger.info("Morning scan started.")

    print("[ 1/2 ]  Unified screener (Momentum -> Claude AI pipeline)...")
    run_unified()

    print("[ 2/2 ]  Chart pattern breakouts...")
    run_chart_patterns()

    print("\n========================================")
    print("  SCAN COMPLETE -- check logs/ for detail")
    print("========================================\n")
