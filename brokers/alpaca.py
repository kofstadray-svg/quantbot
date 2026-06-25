"""
brokers/alpaca.py -- Paper/live order execution via alpaca-py.
"""
from __future__ import annotations
import re
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from loguru import logger
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
from utils.risk import check_position
from utils.retry import with_retry
from utils.tradingview import tv as _tv
from utils.metrics import counter as _counter

_trading = TradingClient(
    ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING
)


def _normalize_crypto_symbol(symbol: str) -> str:
    """Normalize any crypto symbol form to Alpaca's "BASE/QUOTE" format.

    Handles the three forms seen across the codebase and the Alpaca API:
      - "BTC/USD"  (already correct)          -> "BTC/USD"
      - "BTC-USD"  (yfinance format)          -> "BTC/USD"
      - "BTCUSD"   (Alpaca position symbol)   -> "BTC/USD"   ← the bug case
                   ("AVAXUSD" -> "AVAX/USD", "ETHUSDT" -> "ETH/USDT")

    Previously crypto_sell turned "AVAXUSD" into "AVAXUSD/USD" because it only
    replaced a "-USD" suffix and then blindly appended "/USD".
    """
    s = symbol.upper().strip()
    if "/" in s:
        return s
    if "-" in s:                       # yfinance "BTC-USD"
        base, _, quote = s.partition("-")
        return f"{base}/{quote}"
    # No separator (Alpaca position symbol): split a known quote currency off
    # the end. Order matters — check longer suffixes first.
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s                            # leave unchanged if unrecognised


@with_retry(max_attempts=3, base_delay=2.0)
def get_account() -> dict:
    acct = _trading.get_account()
    return {
        "equity":        float(acct.equity),
        "cash":          float(acct.cash),
        "buying_power":  float(acct.buying_power),
    }


@with_retry(max_attempts=3, base_delay=2.0)
def get_positions() -> list[dict]:
    return [
        {
            "symbol":       p.symbol,
            "qty":          float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
        for p in _trading.get_all_positions()
    ]


def _already_holding(symbol: str) -> bool:
    """Return True if there is already an open position in this symbol."""
    try:
        held = {p.symbol.upper() for p in _trading.get_all_positions()}
        return symbol.upper() in held
    except Exception:
        return False


def market_buy(ticker: str, notional_usd: float) -> dict | None:
    """Buy $notional_usd worth of ticker at market price."""
    if _already_holding(ticker):
        logger.info(f"SKIP {ticker} -- already holding a position.")
        return None
    if not check_position(notional_usd):
        return None
    req = MarketOrderRequest(
        symbol=ticker,
        notional=round(notional_usd, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    try:
        order = _trading.submit_order(req)
        logger.info(f"BUY  {ticker}  ${notional_usd:.2f}  order_id={order.id}")
        _counter("trades_placed_total", {"side": "BUY", "market": "stock"})
        _tv.after_trade(ticker)
        return {"id": str(order.id), "status": str(order.status)}
    except Exception as e:
        if "not fractionable" in str(e) or "40310000" in str(e):
            logger.warning(f"SKIP {ticker} -- not fractionable on Alpaca (illiquid/OTC stock).")
        else:
            logger.error(f"BUY {ticker} failed: {e}")
        return None


def market_buy_with_stop(
    ticker: str,
    notional_usd: float,
    stop_pct: float,
    take_profit_pct: float | None = None,
) -> dict | None:
    """
    Buy $notional_usd of ticker and immediately attach a stop-loss (and
    optional take-profit) so the position is protected even between scans.

    stop_pct        : fraction below entry to place stop  (e.g. 0.10 = -10%)
    take_profit_pct : fraction above entry for limit exit (e.g. 0.12 = +12%)

    Uses Alpaca bracket / stop order class so both legs are linked.
    """
    if _already_holding(ticker):
        logger.info(f"SKIP {ticker} -- already holding a position.")
        return None
    if not check_position(notional_usd):
        return None

    # Get current quote to estimate stop / limit levels
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker)
        )
        q = quote.get(ticker) if quote else None
        if q is None:
            logger.warning(f"No quote returned for {ticker} -- falling back to plain market_buy")
            return market_buy(ticker, notional_usd)
        ask_price = float(q.ask_price) or float(q.bid_price)
        bid_price = float(q.bid_price) or ask_price
    except Exception as e:
        logger.warning(f"Could not fetch quote for {ticker}: {e} -- falling back to plain market_buy")
        return market_buy(ticker, notional_usd)

    # Bracket orders require whole shares.
    # If notional < ask_price (e.g. $100 budget, $285 stock), buy 1 share
    # instead of giving up — a protected 1-share position beats a naked buy.
    # Hard-cap: skip if even 1 share would exceed 5× notional (way too expensive).
    import math
    qty = math.floor(notional_usd / ask_price)
    if qty < 1:
        if ask_price > notional_usd * 5:
            logger.warning(
                f"SKIP {ticker} -- price ${ask_price:.2f} is more than 5× "
                f"notional ${notional_usd:.0f}; too expensive for a bracket order."
            )
            return None
        qty = 1
        logger.info(
            f"{ticker} price ${ask_price:.2f} > notional ${notional_usd:.0f} "
            f"-- buying 1 share with bracket protection."
        )

    # ── Stop-loss price ────────────────────────────────────────────────────
    # Alpaca validates the bracket stop against its OWN base_price (the live
    # trade price at submission), not the ask we fetched. For fast-moving names
    # (e.g. AEHR) our fetched ask can be higher/staler than Alpaca's base_price,
    # so a stop that looks fine vs our ask can land ABOVE base_price-0.01 and be
    # rejected. Defend by:
    #   1. anchoring the stop to the conservative bid (<= ask), and
    #   2. requiring a wider mandatory gap (max of $0.05 / 0.5% / stop_pct).
    ref_price   = min(ask_price, bid_price)            # most conservative anchor
    min_gap     = max(0.05, ref_price * 0.005)         # at least 0.5% or 5c below
    raw_stop    = ref_price * (1 - stop_pct)
    stop_price  = round(min(raw_stop, ref_price - min_gap), 2)

    order_class = OrderClass.SIMPLE
    take_profit = False           # flag only; _submit() builds the actual requests
    tp_price    = None

    if take_profit_pct is not None:
        # Must be strictly above ask_price by at least $0.01 AND at least 0.1%.
        raw_tp   = ask_price * (1 + take_profit_pct)
        tp_price = round(max(raw_tp, ask_price + max(0.01, ask_price * 0.001)), 2)
        take_profit = True
        order_class = OrderClass.BRACKET

    def _submit(stop_px, tp_px, oclass):
        sl = StopLossRequest(stop_price=stop_px)
        tp = TakeProfitRequest(limit_price=tp_px) if tp_px is not None else None
        return _trading.submit_order(MarketOrderRequest(
            symbol=ticker, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, order_class=oclass,
            stop_loss=sl, take_profit=tp,
        ))

    try:
        order = _submit(stop_price, tp_price, order_class)
        tp_str = f"  TP=${tp_price:.2f}" if take_profit else ""
        logger.info(
            f"BUY  {ticker}  x{qty} @ ~${ask_price:.2f}  "
            f"stop=${stop_price:.2f}{tp_str}  order_id={order.id}"
        )
        _counter("trades_placed_total", {"side": "BUY", "market": "stock"})
        _tv.after_trade(ticker)
        return {"id": str(order.id), "status": str(order.status)}
    except Exception as e:
        msg = str(e)
        if "not fractionable" in msg or "40310000" in msg:
            logger.warning(f"SKIP {ticker} -- not fractionable on Alpaca.")
            return None

        # Stop/TP rejected against Alpaca's base_price. Alpaca tells us the real
        # base_price in the error — re-anchor the stop (and TP) to it and retry
        # ONCE. This recovers fast-moving names like AEHR where our fetched ask
        # diverged from Alpaca's live base_price between quote and submit.
        if ("base_price" in msg) and re.search(r'"base_price":"?([\d.]+)', msg):
            base = float(re.search(r'"base_price":"?([\d.]+)', msg).group(1))
            new_stop = round(min(base * (1 - stop_pct), base - max(0.05, base * 0.005)), 2)
            new_tp   = None
            if tp_price is not None:
                new_tp = round(max(base * (1 + take_profit_pct), base + max(0.05, base * 0.005)), 2)
            try:
                order = _submit(new_stop, new_tp, order_class)
                tp_str = f"  TP=${new_tp:.2f}" if new_tp is not None else ""
                logger.info(
                    f"BUY  {ticker}  x{qty} @ ~${base:.2f} (re-anchored to base_price)  "
                    f"stop=${new_stop:.2f}{tp_str}  order_id={order.id}"
                )
                _counter("trades_placed_total", {"side": "BUY", "market": "stock"})
                _tv.after_trade(ticker)
                return {"id": str(order.id), "status": str(order.status)}
            except Exception as e2:
                logger.error(
                    f"SKIP {ticker} -- bracket retry vs base_price={base:.2f} still "
                    f"rejected; will NOT place an unprotected buy. Error: {e2}"
                )
                return None

        if "take_profit" in msg or "stop_loss" in msg or "stop_price" in msg:
            # Never silently strip the stop — a naked buy is worse than no trade.
            logger.error(
                f"SKIP {ticker} -- bracket order rejected by Alpaca and will NOT "
                f"fall back to an unprotected buy. Error: {e}"
            )
        else:
            logger.error(f"BUY {ticker} with stop failed: {e}")
        return None


def _exit_tag(reason: str, ticker: str) -> str | None:
    """Build a unique Alpaca client_order_id that encodes the exit reason, so
    the position monitor can classify autonomous exits (supertrend/chandelier/
    time-stop/partial) instead of lumping them all under 'manual'. None if no
    reason given. Alpaca client_order_ids must be unique and <=128 chars."""
    if not reason:
        return None
    import time as _t
    safe = "".join(c for c in reason.lower().replace(" ", "_") if c.isalnum() or c == "_")[:24]
    return f"exit-{safe}-{ticker[:8]}-{int(_t.time()*1000)}"[:128]


def market_sell(ticker: str, qty: float, reason: str = "") -> dict | None:
    kwargs = {}
    tag = _exit_tag(reason, ticker)
    if tag:
        kwargs["client_order_id"] = tag
    req = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        **kwargs,
    )
    order = _trading.submit_order(req)
    logger.info(f"SELL {ticker}  qty={qty}  reason={reason or 'n/a'}  order_id={order.id}")
    _counter("trades_placed_total", {"side": "SELL", "market": "stock"})
    return {"id": str(order.id), "status": str(order.status)}


def crypto_buy(symbol: str, notional_usd: float) -> dict | None:
    """
    Buy $notional_usd worth of a crypto asset.
    Converts yfinance format (BTC-USD) to Alpaca format (BTC/USD) automatically.
    Uses DAY (Alpaca requires DAY for fractional/notional orders; market orders fill instantly anyway).
    """
    alpaca_check = _normalize_crypto_symbol(symbol)
    if _already_holding(alpaca_check):
        logger.info(f"SKIP {alpaca_check} -- already holding a position.")
        return None
    if not check_position(notional_usd):
        return None
    # Normalize symbol: BTC-USD / BTCUSD / BTC -> BTC/USD
    alpaca_symbol = alpaca_check
    req = MarketOrderRequest(
        symbol=alpaca_symbol,
        notional=round(notional_usd, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,  # fractional orders require DAY on Alpaca
    )
    order = _trading.submit_order(req)
    logger.info(f"CRYPTO BUY  {alpaca_symbol}  ${notional_usd:.2f}  order_id={order.id}")
    _counter("trades_placed_total", {"side": "BUY", "market": "crypto"})
    return {"id": str(order.id), "status": str(order.status)}


def crypto_sell(symbol: str, qty: float, reason: str = "") -> dict | None:
    """Sell qty of a crypto asset. DAY order (Alpaca requires DAY for fractional orders).

    Accepts any symbol form (BTC/USD, BTC-USD, or Alpaca's BTCUSD position
    symbol) and normalizes to BASE/QUOTE before submitting. `reason` tags the
    order so the position monitor can classify the exit.
    """
    alpaca_symbol = _normalize_crypto_symbol(symbol)
    kwargs = {}
    tag = _exit_tag(reason, alpaca_symbol.replace("/", ""))
    if tag:
        kwargs["client_order_id"] = tag
    req = MarketOrderRequest(
        symbol=alpaca_symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,  # fractional orders require DAY on Alpaca
        **kwargs,
    )
    order = _trading.submit_order(req)
    logger.info(f"CRYPTO SELL {alpaca_symbol}  qty={qty}  reason={reason or 'n/a'}  order_id={order.id}")
    _counter("trades_placed_total", {"side": "SELL", "market": "crypto"})
    return {"id": str(order.id), "status": str(order.status)}


def close_all_positions() -> None:
    """Emergency: liquidate everything."""
    logger.warning("Closing ALL positions!")
    _trading.close_all_positions(cancel_orders=True)
