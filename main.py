"""
main.py — Entry point. Runs the full bot loop on a schedule.
"""
import schedule
import time
from loguru import logger
from config import validate_config
from utils.risk import reset_daily_loss

# ── logging setup ──────────────────────────────────────────────────────────────
from utils.logging import setup_logging
setup_logging("bot")

# ── config validation + startup banner ───────────────────────────────────────
from config import ENV, IS_PROD, PAPER_TRADING
_msgs = validate_config(die=True)
_mode = "LIVE" if (IS_PROD and not PAPER_TRADING) else "PAPER"
from utils.version import build_stamp as _build_stamp
# ASCII only: unicode box chars crash cp1252 console sinks on Windows, which
# previously blocked the BUILD line from ever logging. BUILD logs FIRST so it
# is always recorded at startup regardless of anything that follows.
logger.info(f"BUILD | {_build_stamp()}")
logger.info(f"=== Trading Bot starting  env={ENV.upper()}  mode={_mode} ===")
for _msg in _msgs:
    if _msg.startswith("ERROR") or _msg.startswith("WARNING"):
        logger.warning(_msg)
    elif not _msg.startswith("INFO"):
        logger.info(_msg)

# ── metrics + health ──────────────────────────────────────────────────────────
from utils.metrics import gauge as _gauge, counter as _counter
from utils.health import HealthServer as _HealthServer
_health = _HealthServer(port=9102, name="bot")
_health.start()
_gauge("bot_up", {}, 1)


def run_stock_screen() -> None:
    from agents.unified_screener import screen
    from brokers.execution import smart_buy, RISK_PER_TRADE_USD
    from utils.candidate_ranker import rank_and_filter
    from data.custom_watchlist import CUSTOM_WATCHLIST
    results = screen(CUSTOM_WATCHLIST)
    for r in results:
        logger.info(f"Screen | {r['ticker']:6s} score={r['score']} total={r.get('total_score',0):+.2f} {r['signal']} [{r.get('regime_label','?')}]")
    # Rank and select top N candidates instead of trading all BUY signals
    top_candidates = rank_and_filter(results)
    for r in top_candidates:
        # risk_usd = base risk budget scaled by regime size multiplier
        # smart_buy converts this to a notional via ATR normalisation
        risk_usd = RISK_PER_TRADE_USD * r.get("size_mult", 1.0)
        logger.info(
            f"Auto-trade triggered: #{r['rank_pos']} {r['ticker']} "
            f"rank={r['rank_score']:.3f} score={r['score']} "
            f"total={r.get('total_score',0):+.2f} "
            f"regime={r.get('regime_label','?')} risk_budget=${risk_usd:.1f}"
        )
        smart_buy(r["ticker"], risk_usd)


def run_candlestick_scan() -> None:
    from agents.candlestick_patterns import screen_candlesticks
    from brokers.alpaca import market_buy
    from data.custom_watchlist import CUSTOM_WATCHLIST
    hits = screen_candlesticks(CUSTOM_WATCHLIST)
    if not hits:
        logger.info("Candlestick scan: no significant patterns today.")
        return
    for h in hits:
        logger.info(
            f"Candle | {h['ticker']:6s} {h['pattern']:30s} {h['action']:4s} "
            f"({h['strength']}) — {h['reason']}"
        )
        # Only auto-buy on strong bullish patterns
        if h["action"] == "BUY" and h["strength"] == "strong":
            logger.info(f"Candle auto-trade → {h['ticker']} ({h['pattern']})")
            market_buy(h["ticker"], 100)


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

    # Bear-regime gate. Walk-forward + regime segmentation showed Chart Pattern
    # is profitable in bull/high-vol (PF ~1.45 OOS) but a money-loser in bear
    # regime (25% win, PF 0.28, -6.7%/trade) — failed breakouts in downtrends.
    # SPY < 50 EMA defines bear. We screen/log patterns either way but BLOCK
    # auto-trade entries while the market regime is bear. Fails open (trades) if
    # the calendar is unavailable, so a data hiccup never silently halts trading.
    bear_blocked = False
    try:
        from datetime import date as _date
        from backtesting.engine import build_regime_calendar, label_regime
        _regime = label_regime(_date.today(), build_regime_calendar())
        if _regime == "bear":
            bear_blocked = True
            logger.warning(
                "Chart pattern: market regime=BEAR (SPY<50EMA) — entries blocked "
                "this scan (strategy loses in bear; see regime backtest). "
                "Patterns still logged below."
            )
        else:
            logger.info(f"Chart pattern: market regime={_regime} — entries allowed.")
    except Exception as _re:
        logger.warning(f"Chart pattern: regime check unavailable ({_re}); allowing entries.")

    from utils.trade_journal import record_signal
    for h in hits:
        logger.info(
            f"Chart  | {h['ticker']:6s} [{h['state']:9s}] {h['pattern']:25s} → {h['action']} — {h['reason']}"
        )
        # Only auto-trade confirmed BUY breakouts — bracket order with hard stop
        if h["state"] == "confirmed" and h["action"] == "BUY":
            if bear_blocked:
                logger.info(f"Chart skip (bear regime) → {h['ticker']} ({h['pattern']})")
                continue
            logger.info(f"Chart auto-trade → {h['ticker']} ({h['pattern']})")
            order = market_buy_with_stop(h["ticker"], 100, stop_pct=0.10, take_profit_pct=0.12)
            if order:
                try:
                    record_signal(
                        ticker=h["ticker"],
                        indicators={"rsi_14": h.get("rsi"), "rel_volume": h.get("rel_volume")},
                        claude_score=0,   # chart-pattern is rule-based
                        entry_price=h.get("price") or h.get("entry"),
                        alpaca_order_id=order.get("id"),
                    )
                except Exception as je:
                    logger.warning(f"Journal | failed to record {h['ticker']}: {je}")


def run_options_flow() -> None:
    from config import OPTIONS_FLOW_ENABLED, UNUSUAL_WHALES_API_KEY
    if not OPTIONS_FLOW_ENABLED:
        logger.debug("Options-flow skipped — UNUSUAL_WHALES_API_KEY not configured.")
        return
    from agents.options_flow import fetch_flow, analyze_flow, execute_flow_trade
    flow = fetch_flow(api_key=UNUSUAL_WHALES_API_KEY)
    if not flow:
        logger.info("No whale flow data available.")
        return
    idea = analyze_flow(flow)
    if idea:
        execute_flow_trade(idea)


def _run_scan(label: str, watchlist: list[str]) -> None:
    """Shared helper — screen a watchlist and auto-trade top-ranked BUY signals."""
    from brokers.execution import smart_buy, RISK_PER_TRADE_USD
    from utils.candidate_ranker import rank_and_filter
    if not watchlist:
        logger.warning(f"{label}: empty watchlist, skipping.")
        return
    logger.info(f"{label}: screening {len(watchlist)} stocks…")
    _counter("scans_run_total", {"scan": label.lower()})
    from agents.unified_screener import screen
    results = screen(watchlist)
    for r in results:
        logger.info(f"{label} | {r['ticker']:6s} score={r['score']} total={r.get('total_score',0):+.2f} {r['signal']} [{r.get('regime_label','?')}]")
    # Rank and select top N candidates instead of trading all BUY signals
    top_candidates = rank_and_filter(results)
    from utils.trade_journal import record_signal
    for r in top_candidates:
        risk_usd = RISK_PER_TRADE_USD * r.get("size_mult", 1.0)
        logger.info(
            f"{label} auto-trade: #{r['rank_pos']} {r['ticker']} "
            f"rank={r['rank_score']:.3f} score={r['score']} "
            f"total={r.get('total_score',0):+.2f} "
            f"regime={r.get('regime_label','?')} risk_budget=${risk_usd:.1f}"
        )
        order = smart_buy(r["ticker"], risk_usd)
        # Record the signal in the trade journal so position_monitor can later
        # reconcile the outcome by alpaca_order_id. Only log on a real fill.
        if order:
            try:
                record_signal(
                    ticker=r["ticker"],
                    indicators={
                        "rsi_14":      r.get("rsi"),
                        "rel_volume":  r.get("rel_volume"),
                        "bb_position": r.get("bb_position"),
                        "macd_signal": r.get("macd_signal"),
                        "golden_cross": r.get("golden_cross"),
                    },
                    claude_score=int(r.get("ai_score") or r.get("score") or 0),
                    entry_price=order.get("limit_price"),
                    alpaca_order_id=order.get("id"),
                )
            except Exception as je:
                logger.warning(f"Journal | failed to record {r['ticker']}: {je}")


def run_tiingo_scan() -> None:
    """
    Tiingo chart-pattern screen — runs independently from the main
    chart_pattern_screener. Adds volume-expansion, ATR-compression,
    trend-alignment scoring and bear-regime veto.
    """
    from agents.tiingo_screener import screen
    from brokers.alpaca import market_buy_with_stop
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist
    from utils.trade_journal import record_signal

    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    results = screen(watchlist)

    if not results:
        logger.info("Tiingo scan: no signals above WATCH threshold.")
        return

    for r in results:
        logger.info(
            f"Tiingo | {r.ticker:6s} [{r.signal_tier:10s}] "
            f"{r.pattern:25s} score={r.score:5.1f}  "
            f"vol={r.rel_vol:.1f}x  ATR={r.atr_ratio:.2f}x  {r.state}"
        )
        # Auto-trade STRONG BUY confirmed signals only
        if r.signal_tier == "STRONG BUY" and r.state == "confirmed" and r.action == "BUY":
            logger.info(f"Tiingo auto-trade → {r.ticker} ({r.pattern})")
            order = market_buy_with_stop(r.ticker, 100, stop_pct=0.10, take_profit_pct=0.12)
            if order:
                try:
                    record_signal(
                        ticker=r.ticker,
                        indicators={"rel_volume": r.rel_vol, "atr_ratio": r.atr_ratio},
                        claude_score=int(r.score),
                        entry_price=None,
                        alpaca_order_id=order.get("id"),
                    )
                except Exception as je:
                    logger.warning(f"Journal | failed to record {r.ticker}: {je}")



def run_small_mid_cap_scan() -> None:
    from data.universe import get_small_and_mid_cap_watchlist
    _run_scan("SmMid", get_small_and_mid_cap_watchlist(top_n=30))


def run_institutional_breakout_scan() -> None:
    """
    Setup B — Institutional Breakout scanner.
    Parallel alpha engine running alongside the existing momentum screener.
    IB wins on conflict (signal_router rule A).
    ATR-based exit: 2 ATR TP1 (50%), 4 ATR TP2 (25%), Chandelier runner (25%).
    """
    from agents.signal_router import route_signals
    from agents.market_regime import compute_regime
    from brokers.alpaca import market_buy_with_stop
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist
    from data.exit_state import tag_strategy         # tags exit_state.json with strategy
    from utils.trade_journal import record_signal

    watchlist = list(dict.fromkeys(
        CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
    ))
    regime = compute_regime()

    # Route through signal_router — IB signals win over momentum on same ticker
    signals = route_signals(watchlist, regime=regime, run_momentum=False, run_ib=True)

    if not signals:
        logger.info("IB scan: no signals this cycle.")
        return

    for sig in signals:
        ticker   = sig["ticker"]
        strategy = sig["strategy"]
        score    = sig["score"]
        entry    = sig["entry"]
        stop     = sig["stop"]
        atr      = sig["atr"]

        logger.info(
            f"IB | {ticker:6s} [{strategy:22s}] score={score:.0f}  "
            f"entry={entry:.2f}  stop={stop:.2f}  ATR={atr:.2f}"
        )

        # Only auto-trade institutional_breakout signals (not momentum — that
        # engine runs via run_stock_screen separately)
        if strategy != "institutional_breakout":
            continue

        logger.info(f"IB auto-trade → {ticker}  entry={entry:.2f}  stop={stop:.2f}")
        # Use fixed $100 notional matching other strategies; ATR stop is tracked
        # in exit_state.json (tagged by strategy) for exit_manager routing.
        order = market_buy_with_stop(ticker, 100, stop_pct=0.10, take_profit_pct=0.20)
        if order:
            # Tag exit_state so exit_manager routes to ATR profile
            try:
                tag_strategy(ticker, strategy=strategy, entry_price=entry, atr=atr)
            except Exception:
                pass
            try:
                record_signal(
                    ticker=ticker,
                    indicators={"rel_volume": 0, "adx": 0},
                    claude_score=int(score),
                    entry_price=entry,
                    alpaca_order_id=order.get("id"),
                    strategy=strategy,
                )
            except Exception as je:
                logger.warning(f"Journal | IB {ticker}: {je}")


def run_nasdaq_scan() -> None:
    from data.universe import get_nasdaq_watchlist
    _run_scan("Nasdaq", get_nasdaq_watchlist(top_n=30))


def run_dow_scan() -> None:
    from data.universe import get_dow_watchlist
    _run_scan("Dow", get_dow_watchlist())


def run_vwap_scan() -> None:
    from agents.vwap_screener import screen, DEFAULT_WATCHLIST
    from data.universe import get_nasdaq_watchlist
    from brokers.alpaca import market_buy
    # Custom watchlist + top Nasdaq movers
    watchlist = list(dict.fromkeys(DEFAULT_WATCHLIST + get_nasdaq_watchlist(top_n=20)))
    signals = screen(watchlist)
    if not signals:
        logger.info("VWAP scan: no setups today.")
        return
    from utils.trade_journal import record_signal
    for s in signals:
        logger.info(
            f"VWAP | {s['ticker']:6s} ${s['price']}  RSI2={s['rsi2']}  "
            f"stop={s['stop']}  target={s['target']}  — {s['reason']}"
        )
        order = market_buy(s["ticker"], 100)
        if order:
            try:
                record_signal(
                    ticker=s["ticker"],
                    indicators={"rsi_14": s.get("rsi2"), "rel_volume": s.get("rel_volume")},
                    claude_score=0,   # VWAP is rule-based, no Claude score
                    entry_price=s.get("price"),
                    alpaca_order_id=order.get("id"),
                )
            except Exception as je:
                logger.warning(f"Journal | failed to record {s['ticker']}: {je}")


def run_crypto_scan() -> None:
    from agents.crypto_mean_reversion import screen
    from brokers.alpaca import crypto_buy
    signals = screen()
    if not signals:
        logger.info("Crypto MeanRev: no setups.")
        return
    from utils.trade_journal import record_signal
    for s in signals:
        logger.info(
            f"Crypto MeanRev | {s['symbol']:<6} ${s['price']}  "
            f"RSI={s['rsi']}  vol={s['rel_volume']}x  "
            f"stop={s['stop']}  target={s['target']}  — {s['reason']}"
        )
        order = crypto_buy(s["symbol"], 100)  # $100 per trade
        if order:
            try:
                record_signal(
                    ticker=s["symbol"],
                    indicators={"rsi_14": s.get("rsi"), "rel_volume": s.get("rel_volume")},
                    claude_score=0,   # crypto mean-reversion is rule-based
                    entry_price=s.get("price"),
                    alpaca_order_id=order.get("id"),
                )
            except Exception as je:
                logger.warning(f"Journal | failed to record {s['symbol']}: {je}")


def run_polymarket() -> None:
    from config import POLYMARKET_ENABLED
    if not POLYMARKET_ENABLED:
        logger.debug("Polymarket skipped — credentials not configured (still placeholder 0x...).")
        return
    from agents.polymarket import fetch_markets, find_edge, size_bets
    markets = fetch_markets(limit=50)
    ideas = find_edge(markets)
    bets = size_bets(ideas)
    for b in bets:
        logger.info(
            f"Polymarket | {b['bet_side']} {b['question'][:55]} "
            f"edge={b['edge']:.1%} ${b['notional_usd']}"
        )


def morning_reset() -> None:
    reset_daily_loss()
    logger.info("=== New trading day started ===")
    _gauge("daily_pnl_usd", {}, 0)
    try:
        from agents.market_regime import compute_regime
        r = compute_regime(force=True)   # fresh fetch at day start
        _gauge("regime_buy_threshold", {}, r.buy_threshold)
        _gauge("regime_size_mult",     {}, r.size_mult)
        _gauge("regime_breadth_mult",  {}, r.breadth_mult)
        _gauge("regime_vix",           {}, r.vix)
        logger.info(
            f"Market regime: {r.label}  "
            f"buy>={r.buy_threshold}  size={r.size_mult}x  breadth={r.breadth_mult}x  "
            f"VIX={r.vix}  ADX={r.adx}  RV={r.realized_vol}%"
        )
        logger.info(f"Breadth: {r.breadth_details}")
    except Exception as _re:
        logger.warning(f"Could not refresh regime at day start: {_re}")


def run_position_monitor() -> None:
    """Poll Alpaca for filled SELL orders and record outcomes in the trade journal."""
    try:
        from utils.position_monitor import run_position_monitor as _monitor
        _monitor()
    except Exception as e:
        logger.error(f"Position monitor error: {e}")


def run_exit_manager() -> None:
    """Evaluate hard stop / time stop / profit ladder / trend exit on all open positions."""
    try:
        from agents.exit_manager import run_exit_manager as _exit
        n = _exit()
        if n:
            logger.info(f"ExitMgr | {n} exit order(s) placed.")
    except Exception as e:
        logger.error(f"Exit manager error: {e}")


def run_risk_check() -> None:
    """Poll live equity and latch a halt if the drawdown limit is breached."""
    try:
        from brokers.alpaca import get_account
        from utils.risk import check_portfolio_drawdown, halt_trading
        from utils.metrics import gauge as _gauge
        acct = get_account()
        equity = float(acct.get("equity", 0) or acct.get("portfolio_value", 0))
        if equity <= 0:
            return
        _gauge("account_equity_usd", {}, equity)
        msg = check_portfolio_drawdown(equity)
        if msg:
            halt_trading(msg)
            # Halt-only by default — exit manager will work existing positions down.
            # Uncomment to also liquidate immediately on breach:
            # from brokers.alpaca import close_all_positions; close_all_positions()
    except Exception as e:
        logger.error(f"Risk check error: {e}")


def run_cancel_unfilled_buys() -> None:
    """
    After the close, cancel any BUY orders that never filled.

    A market BUY left open at the close would otherwise queue and fill at the
    NEXT session's open — at an unknown gap price the signal never accounted for
    (this is what stranded ADPT/AMGN/CTAS overnight). Canceling them keeps entries
    honest: if a signal couldn't get filled during the session it fired in, we
    drop it rather than take a blind overnight gap.

    Open SELL orders are left alone — those are exit/stop legs we WANT to persist.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus, OrderSide
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)

        open_orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, side=OrderSide.BUY, limit=200)
        )
        if not open_orders:
            logger.info("Post-close cleanup: no unfilled BUY orders to cancel.")
            return

        canceled = 0
        for o in open_orders:
            # Only cancel genuinely-unfilled buys (skip partially-filled — those
            # already established a position the exit manager should handle).
            filled = float(getattr(o, "filled_qty", 0) or 0)
            if filled > 0:
                logger.info(
                    f"Post-close: {o.symbol} buy partially filled ({filled}/{o.qty}) "
                    f"— leaving the filled portion, not canceling."
                )
                continue
            try:
                client.cancel_order_by_id(o.id)
                canceled += 1
                logger.info(
                    f"Post-close: canceled unfilled BUY {o.symbol} "
                    f"(qty={o.qty}, status={o.status}, order_id={str(o.id)[:8]})"
                )
            except Exception as ce:
                logger.warning(f"Post-close: could not cancel {o.symbol} buy {str(o.id)[:8]}: {ce}")

        if canceled:
            logger.info(f"Post-close cleanup: canceled {canceled} unfilled BUY order(s).")
    except Exception as e:
        logger.error(f"Post-close cancel-unfilled-buys error: {e}")


# ── schedule ───────────────────────────────────────────────────────────────────
schedule.every().day.at("07:45").do(morning_reset)        # 15 min before the 08:00 morning scan; resets daily loss counter + regime
# Exit manager + position monitor + risk check — every 10 minutes
schedule.every(10).minutes.do(run_exit_manager)
schedule.every(10).minutes.do(run_position_monitor)
schedule.every(10).minutes.do(run_risk_check)
# All times are MACHINE-LOCAL (Mountain). Market hours in MT: open 07:30, close 14:00.
# Morning block: 30 min after open (08:00 MT = 10:00 ET) through ~09:00.
schedule.every().day.at("08:00").do(run_stock_screen)         # 30 min after open — custom watchlist
schedule.every().day.at("08:05").do(run_dow_scan)             # Dow 30
schedule.every().day.at("08:15").do(run_nasdaq_scan)          # Nasdaq 100 top movers
schedule.every().day.at("08:25").do(run_small_mid_cap_scan)   # S&P 400 + S&P 600
schedule.every().day.at("08:40").do(run_vwap_scan)            # VWAP Pullback + RSI(2)
schedule.every().day.at("08:55").do(run_options_flow)
schedule.every().day.at("09:00").do(run_chart_pattern_scan)   # Chart patterns
schedule.every().day.at("09:05").do(run_tiingo_scan)           # Tiingo chart-pattern screen (scored)
schedule.every().day.at("09:10").do(run_institutional_breakout_scan)  # Setup B — IB engine
schedule.every().day.at("11:00").do(run_polymarket)
# Midday block (~11:30 MT = 13:30 ET) — well inside the session.
schedule.every().day.at("11:30").do(run_stock_screen)         # midday: custom watchlist
schedule.every().day.at("11:40").do(run_dow_scan)             # midday: Dow 30
schedule.every().day.at("11:50").do(run_nasdaq_scan)          # midday: Nasdaq 100
schedule.every().day.at("12:00").do(run_small_mid_cap_scan)   # midday: small/mid cap
# Pre-close block — 30 MIN BEFORE CLOSE (13:30 MT = 15:30 ET). Early enough that
# market orders can fill before the 14:00 MT / 16:00 ET bell. NO scans run after
# this; anything later would queue orders to fill at the next open (gap risk).
schedule.every().day.at("13:30").do(run_stock_screen)         # pre-close: custom watchlist
schedule.every().day.at("13:32").do(run_vwap_scan)            # pre-close: VWAP Pullback + RSI(2)
schedule.every().day.at("13:35").do(run_chart_pattern_scan)   # pre-close: chart patterns
schedule.every().day.at("13:40").do(run_tiingo_scan)           # pre-close: Tiingo chart-pattern screen
schedule.every().day.at("13:45").do(run_institutional_breakout_scan)  # pre-close: Setup B — IB engine
# After the close (14:05 MT = 16:05 ET): cancel any BUY orders that never filled
# so they don't sit overnight and fill at an unknown gap price the next open.
schedule.every().day.at("14:05").do(run_cancel_unfilled_buys)
# Crypto -- trades 24/7, scan every 3 hours = 8 scans/day, 7 days a week.
# (interval-based schedule, so it runs on weekends too -- unlike the equity scans)
schedule.every(3).hours.do(run_crypto_scan)
# Autoresearch — generate candidate prompt Monday morning, commit/revert Friday close
schedule.every().monday.at("08:45").do(lambda: __import__("utils.autoresearch", fromlist=["generate_candidate_prompt"]).generate_candidate_prompt())
schedule.every().friday.at("15:45").do(lambda: __import__("utils.autoresearch", fromlist=["compare_and_commit"]).compare_and_commit())


if __name__ == "__main__":
    logger.info("AI Trading Bot starting…  (press Ctrl+C to stop)")

    # ── PREFLIGHT: verify broker + market data + Claude are actually reachable
    # before entering the scheduler loop. validate_config() only checks creds
    # EXIST; this checks they WORK. Fails early (SystemExit) on a broken broker /
    # data feed so we don't run blind. Claude is non-fatal (rule-based strategies
    # still run). Can be relaxed with PREFLIGHT_DIE=0 in the env if desired.
    import os as _os
    from utils.preflight import run_preflight
    _preflight_die = _os.getenv("PREFLIGHT_DIE", "1") != "0"
    try:
        run_preflight(die=_preflight_die, check_claude=True)
    except SystemExit:
        _gauge("bot_up", {}, 0)
        raise

    _health.mark_ready()
    # Run once immediately on startup — wrapped so a crash here doesn't kill the process
    try:
        morning_reset()
    except Exception as _e:
        logger.error(f"Startup morning_reset error: {_e}")
    # Only run the startup scan if we're already past 30-min-after-open. Running
    # pre-market produced NaN gaps / zero rel-vol (incomplete daily bar), so before
    # 08:00 local (10:00 ET) we skip the immediate scan and let the schedule fire
    # it at the proper time. We also skip after the close so a late-night restart
    # doesn't scan stale data.
    from datetime import datetime as _dt, time as _time
    _now_t = _dt.now().time()
    if _time(8, 0) <= _now_t <= _time(13, 5):   # 08:00–13:05 local (MDT) = 10:00 ET–15:05 ET
        try:
            run_stock_screen()
        except Exception as _e:
            logger.error(f"Startup run_stock_screen error: {_e}")
    else:
        logger.info(
            f"Startup scan skipped — {_now_t.strftime('%H:%M')} local is outside "
            f"the 08:00–13:05 window (30 min after open to late afternoon). "
            f"Scheduler will run the morning scan at 08:00."
        )

    _consecutive_errors = 0
    while True:
        try:
            schedule.run_pending()
            _consecutive_errors = 0
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            _gauge("bot_up", {}, 0)
            break
        except Exception as _loop_err:
            _consecutive_errors += 1
            logger.error(f"Scheduler loop error #{_consecutive_errors}: {_loop_err}")
            _counter("scan_errors_total", {"scan": "scheduler"})
            if _consecutive_errors >= 10:
                logger.critical("10 consecutive scheduler errors - exiting for watchdog restart.")
                _gauge("bot_up", {}, 0)
                raise
        time.sleep(30)
