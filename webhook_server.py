"""
webhook_server.py -- TradingView -> Alpaca order bridge.

HOW IT WORKS:
  TradingView fires an alert -> sends a JSON POST to this server ->
  server validates the secret -> executes the order on Alpaca.

SETUP:
  1. Add to your .env:
       WEBHOOK_SECRET=some_long_random_string
       WEBHOOK_PORT=8080

  2. Expose this server to the internet (ngrok recommended):
       ngrok http 8080
       Copy the https://xxxxx.ngrok.io URL

  3. In TradingView -> Alert -> Webhook URL:
       https://xxxxx.ngrok.io/webhook

  4. Alert message templates:
     BUY with stop+target:
     {"ticker":"{{ticker}}","action":"BUY","notional":100,"stop_pct":0.10,"tp_pct":0.12,"secret":"YOUR_SECRET"}

     Simple BUY:
     {"ticker":"{{ticker}}","action":"BUY","notional":100,"secret":"YOUR_SECRET"}

     SELL (close by qty):
     {"ticker":"{{ticker}}","action":"SELL","qty":10,"secret":"YOUR_SECRET"}

     CLOSE (liquidate entire position):
     {"ticker":"{{ticker}}","action":"CLOSE","secret":"YOUR_SECRET"}

SUPPORTED ACTIONS:
  BUY   -- market buy with optional stop_pct / tp_pct bracket
  SELL  -- market sell by qty (number of shares)
  CLOSE -- close the entire open position in that ticker
  SCAN  -- TradingView SUGGESTS a ticker; the bot runs it through its OWN
           momentum screener and only buys if the score clears SCAN_MIN_TIER
           (default "STRONG BUY"). The gatekeeper pattern -- a weak alert can't
           force a trade the bot's model wouldn't take.
           Alert message:
           {"ticker":"{{ticker}}","action":"SCAN","notional":100,"secret":"YOUR_SECRET"}
           Optional: "min_tier":"BUY" to loosen, "stop_pct"/"tp_pct" to override.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from loguru import logger

from config import WEBHOOK_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, FINNHUB_WEBHOOK_SECRET
from brokers.alpaca import market_buy, market_buy_with_stop, market_sell, get_positions

from utils.logging import setup_logging
from utils.retry import with_retry
setup_logging("webhook")

WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))

# SCAN action: which screener tiers are allowed to auto-trade. A TradingView
# alert can SUGGEST a ticker; the bot only buys if its own screener agrees at
# this tier or better. STRONG BUY is the strictest (default); BUY is looser.
SCAN_MIN_TIER = os.getenv("SCAN_MIN_TIER", "STRONG BUY").upper()
_TIER_RANK = {"STRONG BUY": 3, "BUY": 2, "WATCH": 1, "SKIP": 0}

app = Flask(__name__)


# -- Telegram notification helper ----------------------------------------------

@with_retry(max_attempts=3, base_delay=2.0)
def _notify(message: str) -> None:
    """Send a message to Telegram (best-effort, retries up to 3 times)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests as _req
    resp = _req.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
        timeout=5,
    )
    resp.raise_for_status()


# -- Close position helper -----------------------------------------------------

def _close_position(ticker: str) -> dict | None:
    """Sell the entire open position in ticker, if one exists."""
    try:
        positions = get_positions()
        for p in positions:
            if p["symbol"].upper() == ticker.upper():
                qty = p["qty"]
                result = market_sell(ticker, qty)
                return result
        logger.info(f"CLOSE {ticker} -- no open position found.")
        return None
    except Exception as e:
        logger.error(f"CLOSE {ticker} failed: {e}")
        return None


def _scan_and_maybe_buy(ticker: str, notional: float,
                        stop_pct: float, tp_pct: float,
                        min_tier: str | None = None) -> dict:
    """
    SCAN action: TradingView suggests a ticker; the bot runs it through its OWN
    momentum screener and only buys if the score clears the configured tier.
    This makes TradingView a candidate SOURCE and the bot the GATEKEEPER, so a
    weak alert can't drag the bot into a trade its own model wouldn't take.

    Returns a dict describing what happened (always 200-able): either an order
    result, or a "screened_out" verdict with the score so you can see why.
    """
    from agents.momentum_screener import screen

    tier = (min_tier or SCAN_MIN_TIER).upper()
    min_rank = _TIER_RANK.get(tier, 3)

    results = screen([ticker])
    if not results:
        logger.info(f"SCAN {ticker}: screener returned no data (delisted / fetch fail).")
        _notify(f"<b>TradingView SCAN {ticker}</b>\nNo data -- skipped.")
        return {"status": "skipped", "reason": "no_data", "ticker": ticker}

    r = results[0]
    sig   = r["signal"]
    score = r["weighted_score"]
    rank  = _TIER_RANK.get(sig, 0)

    if rank >= min_rank:
        logger.info(f"SCAN {ticker}: {sig} (score {score}) >= {tier} -- BUYING.")
        order = market_buy_with_stop(
            ticker, notional, stop_pct=stop_pct, take_profit_pct=tp_pct
        )
        if order:
            _notify(
                f"<b>TradingView SCAN -> BUY {ticker}</b>\n"
                f"Screener: {sig} (score {score}/100)\n"
                f"Amount: ${notional:.2f}  stop={int(stop_pct*100)}%  tp={int(tp_pct*100)}%\n"
                f"Order ID: <code>{order.get('id','?')}</code>"
            )
            return {"status": "ok", "ticker": ticker, "signal": sig,
                    "score": score, "order": order}
        return {"status": "buy_failed", "ticker": ticker, "signal": sig, "score": score}

    # Screener disagreed -- do NOT trade. Report why.
    logger.info(f"SCAN {ticker}: {sig} (score {score}) < {tier} -- screened out, no trade.")
    _notify(
        f"<b>TradingView SCAN {ticker}</b>\n"
        f"Screener said {sig} (score {score}/100), below {tier} threshold.\n"
        f"RelVol={r['rel_volume']}x  Gap={r['gap_pct']:+.1f}%  RSI={r['rsi']}\n"
        f"No trade taken."
    )
    return {"status": "screened_out", "ticker": ticker,
            "signal": sig, "score": score, "required": tier}


# -- Webhook endpoint ----------------------------------------------------------

@app.route("/webhook", methods=["POST"])
def webhook():
    # Parse JSON body
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}

    if not data:
        logger.warning("Webhook: empty or invalid JSON received.")
        return jsonify({"error": "invalid JSON"}), 400

    # Validate secret
    if WEBHOOK_SECRET:
        incoming_secret = data.get("secret", "")
        if incoming_secret != WEBHOOK_SECRET:
            logger.warning(f"Webhook: invalid secret from {request.remote_addr}")
            return jsonify({"error": "unauthorized"}), 401

    ticker = str(data.get("ticker", "")).upper().strip()
    action = str(data.get("action", "")).upper().strip()

    if not ticker or not action:
        return jsonify({"error": "ticker and action are required"}), 400

    if action not in ("BUY", "SELL", "CLOSE", "SCAN"):
        return jsonify({"error": f"unknown action: {action}"}), 400

    ts = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Webhook [{ts}]: {action} {ticker}  payload={data}")

    # Execute order
    result = None

    if action == "BUY":
        notional = float(data.get("notional", 100))
        stop_pct = data.get("stop_pct")
        tp_pct   = data.get("tp_pct")

        if stop_pct is not None:
            result = market_buy_with_stop(
                ticker, notional,
                stop_pct=float(stop_pct),
                take_profit_pct=float(tp_pct) if tp_pct is not None else None,
            )
        else:
            result = market_buy(ticker, notional)

        if result:
            stop_str = f"  stop={int(float(stop_pct)*100)}%" if stop_pct else ""
            tp_str   = f"  tp={int(float(tp_pct)*100)}%"    if tp_pct   else ""
            _notify(
                f"<b>TradingView -> BUY {ticker}</b>\n"
                f"Amount: ${notional:.2f}{stop_str}{tp_str}\n"
                f"Order ID: <code>{result.get('id','?')}</code>"
            )

    elif action == "SCAN":
        # TradingView suggests; the bot's screener decides.
        notional = float(data.get("notional", 100))
        stop_pct = float(data.get("stop_pct", 0.10))
        tp_pct   = float(data.get("tp_pct", 0.12))
        min_tier = data.get("min_tier")   # optional per-alert override
        result = _scan_and_maybe_buy(ticker, notional, stop_pct, tp_pct, min_tier)
        # SCAN always returns a structured verdict (even when no trade); send it.
        return jsonify(result), 200

    elif action == "SELL":
        qty = float(data.get("qty", 0))
        if qty <= 0:
            return jsonify({"error": "qty must be > 0 for SELL"}), 400
        result = market_sell(ticker, qty)
        if result:
            _notify(f"<b>TradingView -> SELL {ticker}</b>  qty={qty:.4f}")

    elif action == "CLOSE":
        result = _close_position(ticker)
        if result:
            _notify(f"<b>TradingView -> CLOSE {ticker}</b>  (full position)")

    # Response
    if result is None:
        return jsonify({"status": "skipped", "ticker": ticker, "action": action}), 200

    return jsonify({"status": "ok", "ticker": ticker, "action": action, "order": result}), 200


# -- FinnHub webhook endpoint -------------------------------------------------
# FinnHub pushes real-time events to this endpoint when configured.
# Event types handled:
#   news          — company news article published for a watchlist ticker
#   earnings      — actual EPS released (post-market earnings event)
#   recommendation — analyst rating change
#
# Setup:
#   1. Go to finnhub.io → Dashboard → Webhooks → Add Webhook
#   2. URL: https://your-cloudflare-tunnel/finnhub
#   3. Set FINNHUB_WEBHOOK_SECRET in .env (must match what you enter in FinnHub)
#   4. Subscribe to: News, Earnings, Recommendation events
#
# FinnHub sends:  POST /finnhub  with JSON body:
#   {"type": "news", "data": [{...}]}

@app.route("/finnhub", methods=["POST"])
def finnhub_webhook():
    # Validate FinnHub secret (sent as X-Finnhub-Secret header)
    if FINNHUB_WEBHOOK_SECRET:
        incoming = request.headers.get("X-Finnhub-Secret", "")
        if incoming != FINNHUB_WEBHOOK_SECRET:
            logger.warning("FinnHub webhook: invalid secret from {}", request.remote_addr)
            return jsonify({"error": "unauthorized"}), 401

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    event_type = payload.get("type", "unknown")
    data       = payload.get("data", [])

    logger.info("FinnHub webhook: type={} items={}", event_type, len(data) if isinstance(data, list) else 1)

    # Load watchlist for filtering — only act on tickers we care about
    try:
        from data.custom_watchlist import CUSTOM_WATCHLIST
        watchlist = {t.upper() for t in CUSTOM_WATCHLIST}
    except Exception:
        watchlist = set()

    # ── News event ────────────────────────────────────────────────────────────
    if event_type == "news" and isinstance(data, list):
        for article in data[:5]:   # cap at 5 to avoid spam
            ticker   = str(article.get("related", "")).upper().strip()
            headline = article.get("headline", "")
            source   = article.get("source", "")
            url      = article.get("url", "")

            if not ticker or not headline:
                continue

            # Always log; only Telegram-notify for watchlist tickers
            logger.info("FinnHub NEWS | {} | {}", ticker, headline[:120])

            if ticker in watchlist:
                # Fetch composite signal quality for this ticker
                try:
                    from data.finnhub_data import get_signal_quality
                    quality = get_signal_quality(ticker)
                    gate    = quality.get("gate", "PASS")
                    score   = quality.get("composite", 0.0)
                    gate_str = {"PASS": "✅", "WATCH": "⚠️", "BLOCK": "🚫"}.get(gate, "")
                except Exception:
                    gate_str, score, gate = "", 0.0, "PASS"

                _notify(
                    f"📰 <b>FinnHub News — {ticker}</b>\n"
                    f"{headline[:200]}\n"
                    f"Source: {source}\n"
                    f"Signal quality: {gate_str} {gate} ({score:+.2f})\n"
                    f'<a href="{url}">Read more</a>'
                )

    # ── Earnings event ────────────────────────────────────────────────────────
    elif event_type == "earnings" and isinstance(data, list):
        for item in data:
            ticker   = str(item.get("symbol", "")).upper().strip()
            actual   = item.get("actual")
            estimate = item.get("estimate")
            period   = item.get("period", "")

            if not ticker:
                continue

            logger.info("FinnHub EARNINGS | {} | actual={} est={} period={}", ticker, actual, estimate, period)

            if ticker in watchlist:
                if actual is not None and estimate is not None and estimate != 0:
                    surprise = (actual - estimate) / abs(estimate) * 100
                    icon = "🟢" if surprise >= 0 else "🔴"
                    surprise_str = f"{surprise:+.1f}%"
                else:
                    icon, surprise_str = "⚪", "N/A"

                _notify(
                    f"{icon} <b>FinnHub Earnings — {ticker}</b>\n"
                    f"Period: {period}\n"
                    f"Actual EPS: {actual}  |  Estimate: {estimate}\n"
                    f"Surprise: {surprise_str}\n"
                    f"⚡ AVWAP anchor will reset — IB screener will recalculate."
                )

    # ── Recommendation / analyst change ───────────────────────────────────────
    elif event_type == "recommendation" and isinstance(data, list):
        for item in data:
            ticker  = str(item.get("symbol", "")).upper().strip()
            rating  = item.get("rating", "")
            firm    = item.get("firm", "")
            prev    = item.get("previousRating", "")

            if not ticker or ticker not in watchlist:
                continue

            logger.info("FinnHub RECOMMENDATION | {} | {} → {} ({})", ticker, prev, rating, firm)
            _notify(
                f"🏦 <b>FinnHub Rating — {ticker}</b>\n"
                f"{firm}: {prev} → <b>{rating}</b>"
            )

    return jsonify({"status": "ok", "type": event_type}), 200


# -- Health check --------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()}), 200


# -- Main ----------------------------------------------------------------------

def main():
    if not WEBHOOK_SECRET:
        print("\nWARNING: WEBHOOK_SECRET is not set in .env")
        print("   Anyone who knows your URL can place orders!")
        print("   Add:  WEBHOOK_SECRET=some_long_random_string  to your .env\n")
    else:
        print("\nWebhook secret is configured.")

    print(f"\nWebhook server starting on port {WEBHOOK_PORT}")
    print(f"   Local URL : http://localhost:{WEBHOOK_PORT}/webhook")
    print(f"   Health    : http://localhost:{WEBHOOK_PORT}/health")
    print("\nTo expose to TradingView, run in a second terminal:")
    print(f"   ngrok http {WEBHOOK_PORT}")
    print("   Then paste the https://xxxxx.ngrok.io/webhook URL into TradingView\n")
    logger.info(f"Webhook server started on port {WEBHOOK_PORT}.")

    app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False)


if __name__ == "__main__":
    main()
