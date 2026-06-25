"""
brokers/alpaca_options.py — Options order execution via alpaca-py.

Strategy rules (built-in):
  • DTE        : 28–50 days to expiration
  • Strike     : ~2% OTM for calls, ~2% OTM for puts  (targets delta ~0.35–0.45)
  • Delta check: snapshot verified 0.20–0.60 (accepts contract if snapshot unavailable)
  • Qty        : floor(notional / (ask × 100))  — at least 1 contract

NOTE: Options trading must be enabled on your Alpaca account.
  Paper account: log into app.alpaca.markets → paper account → enable options.
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from loguru import logger

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING

_trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_stock_price(ticker: str) -> float | None:
    """Fetch current ask price for the underlying stock."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        quote  = client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker)
        )
        q = quote.get(ticker) if quote else None
        if q is None:
            logger.warning(f"alpaca_options | no quote returned for {ticker}")
            return None
        ask = float(q.ask_price)
        bid = float(q.bid_price)
        return ask if ask > 0 else bid
    except Exception as e:
        logger.warning(f"alpaca_options | could not get price for {ticker}: {e}")
        return None


def _get_option_ask(symbol: str) -> tuple[float | None, float | None]:
    """Fetch (ask_price, delta) for an option contract symbol.

    Always returns a 2-tuple so callers can unpack unconditionally; either
    element may be None when data is unavailable.
    """
    try:
        from alpaca.data.historical import OptionHistoricalDataClient
        from alpaca.data.requests import OptionSnapshotRequest
        client   = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        snapshot = client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=symbol)
        )
        snap = snapshot.get(symbol)
        if snap is None:
            return None, None
        ask   = float(snap.latest_quote.ask_price) if snap.latest_quote else None
        bid   = float(snap.latest_quote.bid_price) if snap.latest_quote else None
        delta = float(snap.greeks.delta)            if snap.greeks      else None
        price = ask if (ask and ask > 0) else bid
        return price, delta
    except Exception as e:
        logger.debug(f"alpaca_options | snapshot unavailable for {symbol}: {e}")
        return None, None


def _find_contract(
    ticker: str,
    contract_type: ContractType,
    strike_multiplier: float,   # 1.02 for calls (2% OTM), 0.98 for puts
) -> tuple[object | None, float | None]:
    """
    Find the best option contract for a ticker:
      - 28–50 DTE
      - Strike closest to current_price × strike_multiplier
      - Delta verified 0.20–0.60 if snapshot available
    Returns (contract, underlying_price) or (None, None).
    """
    price = _get_stock_price(ticker)
    if not price:
        return None, None

    today      = date.today()
    min_exp    = today + timedelta(days=28)
    max_exp    = today + timedelta(days=50)
    target_str = price * strike_multiplier

    # Strike filter: ±15% around target to keep the list small.
    # NOTE: GetOptionContractsRequest requires strike bounds as STRINGS, not
    # floats — passing floats raises a pydantic ValidationError that silently
    # zeroed out every search and made _find_contract always return None.
    str_lo = str(round(price * (strike_multiplier - 0.15), 2))
    str_hi = str(round(price * (strike_multiplier + 0.15), 2))

    try:
        req = GetOptionContractsRequest(
            underlying_symbols   = [ticker],
            status               = "active",
            expiration_date_gte  = str(min_exp),
            expiration_date_lte  = str(max_exp),
            type                 = contract_type,
            strike_price_gte     = str_lo,
            strike_price_lte     = str_hi,
        )
        resp      = _trading.get_option_contracts(req)
        contracts = resp.option_contracts if resp else []
    except Exception as e:
        logger.warning(f"alpaca_options | contract search failed for {ticker}: {e}")
        return None, None

    if not contracts:
        logger.warning(
            f"alpaca_options | no {contract_type.value} contracts found for "
            f"{ticker} (DTE 28-50, strike {str_lo}–{str_hi})"
        )
        return None, None

    # Sort by strike proximity to target
    contracts.sort(key=lambda c: abs(float(c.strike_price) - target_str))

    # Try the top 3 candidates; pick first with acceptable delta
    for contract in contracts[:3]:
        ask, delta = _get_option_ask(contract.symbol)
        if delta is not None:
            if not (0.20 <= abs(delta) <= 0.60):
                logger.debug(
                    f"alpaca_options | {contract.symbol} delta={delta:.2f} "
                    f"out of range — trying next"
                )
                continue
        # Accept: either delta is good, or snapshot unavailable (trust strike proximity)
        logger.info(
            f"alpaca_options | selected {contract.symbol}  "
            f"strike={contract.strike_price}  exp={contract.expiration_date}  "
            f"delta={delta if delta else 'n/a'}  ask={ask if ask else 'n/a'}"
        )
        return contract, price

    # Fallback: just take the closest strike even if delta unknown
    return contracts[0], price


# ── Public order functions ─────────────────────────────────────────────────────

def buy_call(ticker: str, notional_usd: float = 50) -> dict | None:
    """
    Buy a call option on ticker.
    Targets: 28–50 DTE, strike ~2% OTM (delta ~0.40).
    Uses notional_usd to size (minimum 1 contract).
    """
    logger.info(f"OPTIONS | Looking for CALL on {ticker}  notional=${notional_usd}")
    contract, price = _find_contract(ticker, ContractType.CALL, 1.02)
    if not contract:
        return None

    ask, _ = _get_option_ask(contract.symbol)
    if ask and ask > 0:
        qty = max(1, math.floor(notional_usd / (ask * 100)))
    else:
        qty = 1   # fallback: 1 contract

    req = MarketOrderRequest(
        symbol         = contract.symbol,
        qty            = qty,
        side           = OrderSide.BUY,
        time_in_force  = TimeInForce.DAY,
    )
    try:
        order = _trading.submit_order(req)
        logger.info(
            f"OPTIONS CALL BUY  {ticker}  {contract.symbol}  "
            f"qty={qty}  underlying=${price:.2f}  order_id={order.id}"
        )
        return {
            "id":       str(order.id),
            "symbol":   contract.symbol,
            "ticker":   ticker,
            "type":     "CALL",
            "qty":      qty,
            "strike":   str(contract.strike_price),
            "expiry":   str(contract.expiration_date),
        }
    except Exception as e:
        logger.error(f"OPTIONS CALL BUY {ticker} failed: {e}")
        return None


def buy_put(ticker: str, notional_usd: float = 50) -> dict | None:
    """
    Buy a put option on ticker.
    Targets: 28–50 DTE, strike ~2% OTM (delta ~0.40).
    Uses notional_usd to size (minimum 1 contract).
    """
    logger.info(f"OPTIONS | Looking for PUT on {ticker}  notional=${notional_usd}")
    contract, price = _find_contract(ticker, ContractType.PUT, 0.98)
    if not contract:
        return None

    ask, _ = _get_option_ask(contract.symbol)
    if ask and ask > 0:
        qty = max(1, math.floor(notional_usd / (ask * 100)))
    else:
        qty = 1

    req = MarketOrderRequest(
        symbol         = contract.symbol,
        qty            = qty,
        side           = OrderSide.BUY,
        time_in_force  = TimeInForce.DAY,
    )
    try:
        order = _trading.submit_order(req)
        logger.info(
            f"OPTIONS PUT  BUY  {ticker}  {contract.symbol}  "
            f"qty={qty}  underlying=${price:.2f}  order_id={order.id}"
        )
        return {
            "id":       str(order.id),
            "symbol":   contract.symbol,
            "ticker":   ticker,
            "type":     "PUT",
            "qty":      qty,
            "strike":   str(contract.strike_price),
            "expiry":   str(contract.expiration_date),
        }
    except Exception as e:
        logger.error(f"OPTIONS PUT BUY {ticker} failed: {e}")
        return None


def get_options_positions() -> list[dict]:
    """Return all open options positions."""
    try:
        positions = _trading.get_all_positions()
        opts = []
        for p in positions:
            sym = p.symbol
            # Options symbols are long strings like AAPL240119C00185000
            if len(sym) > 6:
                opts.append({
                    "symbol":       sym,
                    "qty":          float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "cost_basis":   float(p.cost_basis),
                })
        return opts
    except Exception as e:
        logger.error(f"get_options_positions failed: {e}")
        return []


def close_option(symbol: str) -> dict | None:
    """Close (sell to close) an entire options position by contract symbol."""
    try:
        positions = _trading.get_all_positions()
        for p in positions:
            if p.symbol.upper() == symbol.upper():
                qty = float(p.qty)
                req = MarketOrderRequest(
                    symbol        = symbol,
                    qty           = qty,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
                order = _trading.submit_order(req)
                logger.info(f"OPTIONS CLOSE {symbol}  qty={qty}  order_id={order.id}")
                return {"id": str(order.id), "symbol": symbol, "qty": qty}
        logger.warning(f"close_option: no open position found for {symbol}")
        return None
    except Exception as e:
        logger.error(f"close_option {symbol} failed: {e}")
        return None
