"""
utils/position_monitor.py — Automated trade outcome recorder.

Polls Alpaca for recently filled SELL orders and matches them to open
journal entries, then calls record_outcome() so the autoresearch loop
has real win/loss data.

HOW IT WORKS:
  1. Fetch all closed SELL orders from Alpaca (last 7 days by default)
  2. Load open (unresolved) signals from the trade journal
  3. Match by ticker: if a filled SELL exists for a ticker with an open
     signal, compute return_pct from (fill_price / entry_price - 1)
  4. Detect exit_reason from order metadata:
       - stop_loss leg of bracket → "stop"
       - take_profit (limit) leg → "target"
       - plain market sell        → "manual"
  5. Call record_outcome() for each match

SCHEDULING:
  Called from main.py every 10 minutes during market hours:
    schedule.every(10).minutes.do(run_position_monitor)
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING

# Lazy import so missing config doesn't break other modules at startup
def _get_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)


def _get_open_signals() -> list[dict]:
    """Return journal signals with no outcome yet (profitable IS NULL)."""
    from utils.trade_journal import _conn
    with _conn() as con:
        rows = con.execute("""
            SELECT id, ticker, signal_date, entry_price, alpaca_order_id
              FROM signals
             WHERE profitable IS NULL
             ORDER BY signal_date DESC
        """).fetchall()
    return [dict(r) for r in rows]


def _fetch_filled_sells(lookback_days: int = 7) -> list[dict]:
    """
    Fetch filled SELL orders from Alpaca for the last lookback_days.
    Returns list of dicts with: ticker, fill_price, filled_at, order_type,
    order_id, parent_order_id.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderSide

    client = _get_trading_client()
    after = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    try:
        orders = client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                side=OrderSide.SELL,
                after=after,
                limit=200,
            )
        )
    except Exception as e:
        logger.error(f"PositionMonitor | failed to fetch orders: {e}")
        return []

    result = []
    for o in orders:
        # Only process filled orders with a known fill price
        if str(o.status) not in ("filled", "OrderStatus.FILLED"):
            continue
        filled_at = o.filled_at or o.updated_at
        fill_price = float(o.filled_avg_price or 0)
        if fill_price <= 0:
            continue

        # Determine exit reason. Priority:
        #   1. our client_order_id tag (exit-<reason>-<ticker>-<ts>) — set by the
        #      exit manager so autonomous exits are correctly labeled
        #      (supertrend_flip / chandelier / hard_stop / time_stop / partial /
        #      trail / runner_chandelier) instead of collapsing to "manual".
        #   2. order type (stop/limit legs of a bracket)
        #   3. fallback "manual" (a sell with no tag and no bracket = hand close)
        coid = str(getattr(o, "client_order_id", "") or "")
        order_type = str(o.order_type).lower()

        if coid.startswith("exit-"):
            parts = coid.split("-")
            reason = parts[1] if len(parts) >= 2 and parts[1] else "auto_exit"
        elif "stop" in order_type:
            reason = "stop"
        elif "limit" in order_type and "stop" not in order_type:
            reason = "target"
        else:
            reason = "manual"

        result.append({
            "ticker":          o.symbol,
            "fill_price":      fill_price,
            "filled_at":       filled_at,
            "exit_reason":     reason,
            "order_id":        str(o.id),
            "client_order_id": str(getattr(o, "client_order_id", "") or ""),
            "qty":             float(o.filled_qty or o.qty or 0),
        })

    logger.debug(f"PositionMonitor | found {len(result)} filled SELL orders (last {lookback_days}d)")
    return result


def _recover_entry_price(order_id: str | None) -> tuple[float | None, str]:
    """
    Recover a missing entry price from the BUY order's actual fill on Alpaca.

    Returns (fill_price, status):
      - (price, "filled")    -> recovered the real fill price
      - (None, "pending")    -> order accepted/new/partially_filled but no avg
                                price yet; NOT an error, just not filled. Caller
                                should skip quietly and retry next cycle.
      - (None, "unknown")    -> order not found / lookup failed -> real problem.
    Every journaled signal stores its order_id, so this makes the monitor
    self-healing once the order actually fills.
    """
    if not order_id:
        return None, "unknown"
    try:
        client = _get_trading_client()
        o = client.get_order_by_id(order_id)
        fp = getattr(o, "filled_avg_price", None)
        if fp:
            return float(fp), "filled"
        # No fill price yet — is the order still working, or dead?
        status = str(getattr(o, "status", "")).lower()
        if any(s in status for s in ("accepted", "new", "pending", "partially", "held", "calculated")):
            return None, "pending"
        return None, "unknown"   # canceled/rejected/expired with no fill
    except Exception as e:
        logger.debug(f"PositionMonitor | could not recover fill for order {order_id}: {e}")
        return None, "unknown"


def run_position_monitor(lookback_days: int = 7) -> int:
    """
    Main entry point.  Matches filled SELL orders to open journal signals
    and records outcomes.  Returns number of outcomes recorded.
    """
    from utils.trade_journal import record_outcome, _conn

    open_signals = _get_open_signals()
    if not open_signals:
        logger.debug("PositionMonitor | no open journal signals to resolve.")
        return 0

    filled_sells = _fetch_filled_sells(lookback_days)
    if not filled_sells:
        logger.debug("PositionMonitor | no filled SELL orders found.")
        return 0

    # Build a lookup: ticker → list of fill events (could be multiple sells)
    from collections import defaultdict
    sells_by_ticker: dict[str, list[dict]] = defaultdict(list)
    for s in filled_sells:
        sells_by_ticker[s["ticker"]].append(s)

    # Match open signals to fills. Resolve at most ONE open signal per ticker
    # per monitor run, using the OLDEST open signal first -- prevents one SELL
    # fill from stamping the same return onto several duplicate rows (the bug
    # that inflated the track record). Once a ticker is matched, drop it.
    open_signals = sorted(open_signals, key=lambda s: (s.get("signal_date") or "", s["id"]))
    recorded = 0
    resolved_tickers: set[str] = set()
    for sig in open_signals:
        ticker       = sig["ticker"]
        entry_price  = sig.get("entry_price")
        signal_id    = sig["id"]
        signal_date  = sig.get("signal_date", "")

        if ticker in resolved_tickers:
            continue  # already resolved one signal for this ticker this run

        if ticker not in sells_by_ticker:
            continue  # position still open

        if not entry_price or entry_price <= 0:
            # No entry price stored — recover it from the BUY order's actual
            # fill price via Alpaca, then backfill the journal so this self-heals.
            recovered, status = _recover_entry_price(sig.get("alpaca_order_id"))
            if recovered:
                entry_price = recovered
                try:
                    from utils.trade_journal import _conn
                    with _conn() as con:
                        con.execute("UPDATE signals SET entry_price=? WHERE id=?",
                                    (entry_price, signal_id))
                    logger.info(
                        f"PositionMonitor | #{signal_id} ({ticker}) entry_price was "
                        f"missing — recovered {entry_price:.2f} from Alpaca fill, backfilled."
                    )
                except Exception as e:
                    logger.debug(f"PositionMonitor | backfill write failed for #{signal_id}: {e}")
            elif status == "pending":
                # BUY order accepted but not filled yet (e.g. placed after hours,
                # queued for next open). Not an error — quietly skip; it will
                # resolve once the order fills. Avoids warning spam every cycle.
                logger.debug(
                    f"PositionMonitor | #{signal_id} ({ticker}) buy order not filled "
                    f"yet (pending) — will resolve after fill."
                )
                continue
            else:
                logger.warning(
                    f"PositionMonitor | signal #{signal_id} ({ticker}) has no entry_price "
                    f"and order {sig.get('alpaca_order_id','?')} is not recoverable "
                    f"(canceled/rejected/missing) — marking abandoned."
                )
                # Self-heal: mark as abandoned so it never warns again.
                try:
                    from datetime import date as _date
                    with _conn() as con:
                        con.execute(
                            "UPDATE signals SET profitable=0, exit_date=?, "
                            "return_pct=0.0, exit_reason='abandoned_no_fill' WHERE id=?",
                            (_date.today().isoformat(), signal_id),
                        )
                except Exception as _je:
                    logger.debug(f"PositionMonitor | could not mark #{signal_id} abandoned: {_je}")
                continue

        # Pick the most recent fill for this ticker (in case of partial fills)
        fills = sorted(sells_by_ticker[ticker], key=lambda x: x["filled_at"] or "", reverse=True)
        fill = fills[0]

        return_pct = (fill["fill_price"] / entry_price) - 1.0

        # Feed realized dollar losses into the daily-loss circuit breaker.
        qty = fill.get("qty", 0.0)
        dollar_pnl = (fill["fill_price"] - entry_price) * qty
        if dollar_pnl < 0:
            from utils.risk import record_loss
            record_loss(-dollar_pnl)

        exit_date_str = None
        if fill["filled_at"]:
            try:
                exit_date_str = fill["filled_at"].date().isoformat()
            except AttributeError:
                exit_date_str = str(fill["filled_at"])[:10]

        from datetime import date
        exit_date = date.fromisoformat(exit_date_str) if exit_date_str else date.today()

        record_outcome(
            signal_id=signal_id,
            return_pct=return_pct,
            exit_reason=fill["exit_reason"],
            exit_date=exit_date,
        )
        logger.info(
            f"PositionMonitor | resolved #{signal_id} {ticker}  "
            f"entry={entry_price:.2f}  exit={fill['fill_price']:.2f}  "
            f"return={return_pct:+.2%}  reason={fill['exit_reason']}"
        )
        recorded += 1
        resolved_tickers.add(ticker)

    if recorded:
        logger.info(f"PositionMonitor | recorded {recorded} outcome(s).")
    return recorded


if __name__ == "__main__":
    # Quick manual test
    n = run_position_monitor(lookback_days=30)
    print(f"Resolved {n} open signals.")
