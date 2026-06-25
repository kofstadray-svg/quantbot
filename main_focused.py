"""
main_focused.py — Focused bot: Nasdaq, Dow Jones, and Crypto only.

STRATEGIES:
  Stocks  — Claude AI screener, BUY when score >= 8
            (RSI, MACD, golden cross, relative volume, Bollinger Bands,
             Stochastic, OBV trend, Volume RSI)
  Crypto  — Mean Reversion: buy at lower Bollinger Band + RSI < 40 + volume spike

SCHEDULE:
  09:00  Daily loss reset
  09:30  Nasdaq 100 scan
  09:30  Dow 30 scan
  09:30  Chart pattern scan
  09:30  Crypto mean reversion scan

HOW TO RUN:
  python main_focused.py
"""
import schedule
import time
from loguru import logger
from utils.risk import reset_daily_loss

# ── Logging ────────────────────────────────────────────────────────────────────
from utils.logging import setup_logging
setup_logging("focused")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stock_scan(label: str, watchlist: list[str]) -> None:
    """Screen a watchlist with Claude and auto-trade BUY signals (score >= 8)."""
    from agents.stock_screener import screen
    from brokers.alpaca import market_buy

    if not watchlist:
        logger.warning(f"{label}: empty watchlist, skipping.")
        return

    logger.info(f"{label}: scanning {len(watchlist)} stocks...")
    results = screen(watchlist)
    for r in results:
        logger.info(
            f"{label} | {r['ticker']:6s} score={r['score']} {r['signal']} — {r['reason']}"
        )
        if r.get("signal") == "BUY" and r.get("score", 0) >= 8:
            logger.info(f"{label} auto-trade -> {r['ticker']} (score={r['score']})")
            market_buy(r["ticker"], 100)


# ── Scan functions ─────────────────────────────────────────────────────────────

def run_nasdaq_scan() -> None:
    from data.universe import get_nasdaq_watchlist
    _stock_scan("Nasdaq", get_nasdaq_watchlist(top_n=50))


def run_dow_scan() -> None:
    from data.universe import get_dow_watchlist, get_nasdaq_watchlist
    dow = get_dow_watchlist()
    nasdaq_extra = [t for t in get_nasdaq_watchlist(top_n=50) if t not in dow]
    _stock_scan("Dow", dow + nasdaq_extra[:20])


def run_chart_pattern_scan() -> None:
    from agents.chart_pattern_screener import screen_chart_patterns
    from brokers.alpaca import market_buy_with_stop
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist

    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    hits = screen_chart_patterns(watchlist)
    if not hits:
        logger.info("Chart pattern scan: no patterns detected.")
        return
    for h in hits:
        logger.info(
            f"Chart  | {h['ticker']:6s} [{h['state']:9s}] {h['pattern']:25s} -> {h['action']} — {h['reason']}"
        )
        if h["state"] == "confirmed" and h["action"] == "BUY":
            logger.info(f"Chart auto-trade -> {h['ticker']} ({h['pattern']})")
            market_buy_with_stop(h["ticker"], 100, stop_pct=0.10, take_profit_pct=0.12)


def run_crypto_scan() -> None:
    from agents.crypto_mean_reversion import screen
    from brokers.alpaca import crypto_buy

    signals = screen()
    if not signals:
        logger.info("Crypto MeanRev: no setups.")
        return
    for s in signals:
        logger.info(
            f"Crypto MeanRev | {s['symbol']:<6} ${s['price']}  "
            f"RSI={s['rsi']}  vol={s['rel_volume']}x  "
            f"stop={s['stop']}  target={s['target']}"
        )
        crypto_buy(s["symbol"], 100)


def morning_reset() -> None:
    reset_daily_loss()
    logger.info("=== New trading day started ===")


# ── Schedule ───────────────────────────────────────────────────────────────────
schedule.every().day.at("09:00").do(morning_reset)
schedule.every().day.at("09:30").do(run_nasdaq_scan)        # 9:30 AM Nasdaq
schedule.every().day.at("09:30").do(run_dow_scan)           # 9:30 AM Dow
schedule.every().day.at("09:30").do(run_chart_pattern_scan) # 9:30 AM chart patterns
schedule.every().day.at("09:30").do(run_crypto_scan)        # 9:30 AM crypto


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Focused Bot starting...  (Nasdaq + Dow + Crypto Mean Reversion)")
    logger.info("Press Ctrl+C to stop.")

    morning_reset()
    run_crypto_scan()   # run crypto once immediately on startup

    while True:
        schedule.run_pending()
        time.sleep(30)
