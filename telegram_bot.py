"""
telegram_bot.py — Control your trading bot from your phone via Telegram.

FIRST-TIME SETUP:
  1. Open Telegram → search @BotFather → send /newbot → follow prompts
  2. Copy the token BotFather gives you
  3. Add to your .env:
       TELEGRAM_BOT_TOKEN=<your_token>
  4. Run this script:  python telegram_bot.py
  5. Open Telegram, message your new bot /start
  6. Copy the Chat ID it prints, then add to your .env:
       TELEGRAM_CHAT_ID=<your_chat_id>
  7. Restart: python telegram_bot.py  — you're live!

COMMANDS (from your phone):
  /account    — equity, cash, buying power
  /positions  — all open positions + unrealised P&L
  /scan       — full morning scan (Nasdaq + Dow + Chart + Crypto)
  /nasdaq     — Nasdaq 100 screener only
  /dow        — Dow 30 screener only
  /crypto     — Crypto mean reversion scan only
  /add TICKER — add a ticker to your watchlist
  /remove TICKER — remove a ticker from your watchlist
  /watchlist  — show current watchlist
  /help       — list all commands
"""
from __future__ import annotations
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logging import setup_logging
from utils.health import HealthServer

setup_logging("telegram")
_health = HealthServer(port=9100, name="telegram-bot")
_health.start()

_executor = ThreadPoolExecutor(max_workers=2)


# ── Security: only respond to the configured chat ──────────────────────────────

def _authorised(update: Update) -> bool:
    cid = str(update.effective_chat.id)
    if not TELEGRAM_CHAT_ID:
        return True          # not configured yet — allow /start so they can get their ID
    return cid == str(TELEGRAM_CHAT_ID)


async def _deny(update: Update) -> None:
    await update.message.reply_text("⛔ Unauthorised.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_in_thread(fn, *args):
    """Run a blocking function in a thread and return the result."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, fn, *args)


def _fmt_positions(positions: list[dict]) -> str:
    if not positions:
        return "No open positions."
    lines = ["<pre>"]
    lines.append(f"{'Symbol':<6}  {'MktVal':>8}  {'P&L':>8}")
    lines.append("-" * 28)
    total = 0.0
    for p in positions:
        pl = p["unrealized_pl"]
        total += pl
        sign = "+" if pl >= 0 else ""
        lines.append(f"{p['symbol']:<6}  ${p['market_value']:>7,.2f}  {sign}{pl:>7.2f}")
    lines.append("-" * 28)
    sign = "+" if total >= 0 else ""
    lines.append(f"{'TOTAL':>17}  {sign}{total:>7.2f}")
    lines.append("</pre>")
    return "\n".join(lines)


# ── Command handlers ───────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    msg = (
        f"👋 <b>Trading Bot connected!</b>\n\n"
        f"Your Chat ID: <code>{cid}</code>\n\n"
        "Add <code>TELEGRAM_CHAT_ID=" + str(cid) + "</code> to your .env, "
        "then restart this script.\n\n"
        "Send /help to see all commands."
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    msg = (
        "<b>📊 Trading Bot Commands</b>\n\n"
        "/account — portfolio summary\n"
        "/positions — open stock positions &amp; P&amp;L\n"
        "/options — open options positions\n"
        "/scan — full morning scan\n"
        "/nasdaq — Nasdaq screener\n"
        "/dow — Dow screener\n"
        "/tiingo — Tiingo chart-pattern screen (scored)\n"
        "/ib — Institutional Breakout scanner (Setup B)\n"
        "/perf — strategy attribution (momentum vs IB win rates)\n"
        "/research — indicator win rates + autoresearch status\n"
        "/add TICKER — add to watchlist\n"
        "/remove TICKER — remove from watchlist\n"
        "/watchlist — show watchlist\n"
        "/help — this message"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_account(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("Fetching account…")

    def _fetch():
        from brokers.alpaca import get_account
        return get_account()

    acct = await _run_in_thread(_fetch)
    msg = (
        "<b>💰 Paper Account</b>\n\n"
        f"Equity:       <code>${acct['equity']:>12,.2f}</code>\n"
        f"Cash:         <code>${acct['cash']:>12,.2f}</code>\n"
        f"Buying Power: <code>${acct['buying_power']:>12,.2f}</code>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("Fetching positions…")

    def _fetch():
        from brokers.alpaca import get_positions
        return get_positions()

    positions = await _run_in_thread(_fetch)
    await update.message.reply_text(
        f"<b>📈 Open Positions ({len(positions)})</b>\n\n" + _fmt_positions(positions),
        parse_mode="HTML"
    )


async def cmd_options(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("Fetching options positions…")

    def _fetch():
        from brokers.alpaca_options import get_options_positions
        return get_options_positions()

    positions = await _run_in_thread(_fetch)

    if not positions:
        await update.message.reply_text("No open options positions.")
        return

    lines = [f"<b>🎯 Options Positions ({len(positions)})</b>\n", "<pre>"]
    lines.append(f"{'Symbol':<22}  {'Qty':>3}  {'MktVal':>8}  {'P&L':>8}")
    lines.append("-" * 48)
    total_pl = 0.0
    for p in positions:
        pl    = p["unrealized_pl"]
        total_pl += pl
        sign  = "+" if pl >= 0 else ""
        lines.append(
            f"{p['symbol']:<22}  {int(p['qty']):>3}  "
            f"${p['market_value']:>7,.2f}  {sign}{pl:>7.2f}"
        )
    lines.append("-" * 48)
    sign = "+" if total_pl >= 0 else ""
    lines.append(f"{'TOTAL':>29}  {sign}{total_pl:>7.2f}")
    lines.append("</pre>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    logger.info(f"Telegram: /scan received from chat {update.effective_chat.id}")
    await update.message.reply_text(
        "🔍 <b>Scanning…</b> ranking the full watchlist by score. "
        "~30–60s on a fresh run.",
        parse_mode="HTML"
    )

    def _run():
        # Lightweight, ALWAYS-reports version: run ONLY the momentum screener
        # (the full-visibility board). No second unified_screen re-scan, no
        # Claude calls, no chart stage — those made the handler take minutes and
        # silently time out. Trades still happen on the SCHEDULED scans; /scan
        # here is for visibility on demand.
        from agents.momentum_screener import screen as momentum_screen
        from data.custom_watchlist import CUSTOM_WATCHLIST
        from data.universe import get_nasdaq_watchlist, get_dow_watchlist
        from collections import Counter

        summary = {"scanned": 0, "board": [], "tier_counts": {}, "error": None}
        try:
            watchlist = list(dict.fromkeys(
                CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
            ))
            mom = momentum_screen(watchlist)
            summary["scanned"] = len(mom)
            summary["tier_counts"] = dict(Counter(r["signal"] for r in mom))
            for r in mom[:15]:
                summary["board"].append({
                    "t": r["ticker"], "score": r["weighted_score"], "sig": r["signal"],
                    "rv": r.get("rel_volume"), "gap": r.get("gap_pct"), "rsi": r.get("rsi"),
                })
        except Exception as e:
            import traceback
            summary["error"] = f"{type(e).__name__}: {e}"
            logger.error(f"/scan failed: {e}\n{traceback.format_exc()}")
        return summary

    try:
        s = await _run_in_thread(_run)
    except Exception as e:
        logger.error(f"/scan thread error: {e}")
        await update.message.reply_text(f"⚠️ Scan failed: {e}")
        return

    if s["error"]:
        await update.message.reply_text(f"⚠️ Scan error: {s['error']}")
        return

    lines = ["<b>📊 Scan Complete</b>"]
    tc = s.get("tier_counts", {})
    lines.append(f"Scanned <b>{s['scanned']}</b> stocks")
    lines.append(
        f"STRONG BUY {tc.get('STRONG BUY',0)} · BUY {tc.get('BUY',0)} · "
        f"WATCH {tc.get('WATCH',0)} · SKIP {tc.get('SKIP',0)}"
    )
    lines.append("")
    if s["board"]:
        lines.append("<b>🏆 Top candidates by score</b>")
        lines.append("<pre>")
        lines.append(f"{'Tkr':<6}{'Scr':>4} {'Sig':<10}{'RelV':>5}{'Gap%':>6}{'RSI':>5}")
        for b in s["board"]:
            rv  = f"{b['rv']:.1f}" if isinstance(b['rv'], (int,float)) else "-"
            gap = f"{b['gap']:+.1f}" if isinstance(b['gap'], (int,float)) else "-"
            rsi = f"{b['rsi']:.0f}" if isinstance(b['rsi'], (int,float)) else "-"
            lines.append(f"{b['t']:<6}{b['score']:>4.0f} {b['sig']:<10}{rv:>5}{gap:>6}{rsi:>5}")
        lines.append("</pre>")
    else:
        lines.append("No stocks returned data.")

    msg = "\n".join(lines)
    try:
        await update.message.reply_text(msg, parse_mode="HTML")
        logger.info(f"Telegram: /scan reply sent ({s['scanned']} scanned)")
    except Exception as e:
        # HTML parse failures are a classic silent-no-reply cause — retry plain.
        logger.warning(f"/scan HTML reply failed ({e}); sending plain text.")
        plain = msg.replace("<b>","").replace("</b>","").replace("<pre>","").replace("</pre>","")
        await update.message.reply_text(plain[:4000])


async def cmd_nasdaq(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("🔍 Running Nasdaq scan…")

    def _run():
        from agents.stock_screener import screen
        from data.universe import get_nasdaq_watchlist
        results = screen(get_nasdaq_watchlist(top_n=50))
        buys = [r for r in results if r.get("signal") == "BUY" and r.get("score", 0) >= 8]
        return results, buys

    results, buys = await _run_in_thread(_run)
    if not buys:
        await update.message.reply_text(f"Nasdaq scan done — no BUY signals (score ≥8) from {len(results)} stocks.")
    else:
        lines = [f"<b>Nasdaq BUY signals ({len(buys)})</b>\n"]
        for r in buys:
            lines.append(f"• <b>{r['ticker']}</b>  score={r['score']}  {r['reason']}")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_dow(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("🔍 Running Dow scan…")

    def _run():
        from agents.stock_screener import screen
        from data.universe import get_dow_watchlist
        results = screen(get_dow_watchlist())
        buys = [r for r in results if r.get("signal") == "BUY" and r.get("score", 0) >= 8]
        return results, buys

    results, buys = await _run_in_thread(_run)
    if not buys:
        await update.message.reply_text(f"Dow scan done — no BUY signals (score ≥8) from {len(results)} stocks.")
    else:
        lines = [f"<b>Dow BUY signals ({len(buys)})</b>\n"]
        for r in buys:
            lines.append(f"• <b>{r['ticker']}</b>  score={r['score']}  {r['reason']}")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")



async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    if not ctx.args:
        await update.message.reply_text("Usage: /add TICKER")
        return
    ticker = ctx.args[0].upper().strip()

    watchlist_path = os.path.join(os.path.dirname(__file__), "data", "custom_watchlist.py")
    with open(watchlist_path, "r") as f:
        content = f.read()

    if f'"{ticker}"' in content:
        await update.message.reply_text(f"{ticker} is already in your watchlist.")
        return

    new_content = content.rstrip()
    # Insert before closing bracket
    if new_content.endswith("]"):
        new_content = new_content[:-1].rstrip().rstrip(",") + f', "{ticker}",\n]'
    else:
        new_content = new_content + f'\n    "{ticker}",\n]'

    with open(watchlist_path, "w") as f:
        f.write(new_content)

    await update.message.reply_text(f"✅ <b>{ticker}</b> added to your watchlist.", parse_mode="HTML")
    logger.info(f"Telegram: added {ticker} to watchlist.")


async def cmd_remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    if not ctx.args:
        await update.message.reply_text("Usage: /remove TICKER")
        return
    ticker = ctx.args[0].upper().strip()

    watchlist_path = os.path.join(os.path.dirname(__file__), "data", "custom_watchlist.py")
    with open(watchlist_path, "r") as f:
        content = f.read()

    if f'"{ticker}"' not in content:
        await update.message.reply_text(f"{ticker} is not in your watchlist.")
        return

    import re
    new_content = re.sub(rf',?\s*"{re.escape(ticker)}"\s*,?', lambda m: "," if m.group().count(",") == 2 else "", content)
    new_content = re.sub(r',\s*,', ",", new_content)  # clean double commas

    with open(watchlist_path, "w") as f:
        f.write(new_content)

    await update.message.reply_text(f"🗑️ <b>{ticker}</b> removed from your watchlist.", parse_mode="HTML")
    logger.info(f"Telegram: removed {ticker} from watchlist.")


async def cmd_watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)

    watchlist_path = os.path.join(os.path.dirname(__file__), "data", "custom_watchlist.py")
    with open(watchlist_path, "r") as f:
        content = f.read()

    import re
    tickers = re.findall(r'"([A-Z]+)"', content)
    msg = f"<b>📋 Watchlist ({len(tickers)} tickers)</b>\n\n" + "  ".join(tickers)
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_tiingo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the Tiingo chart-pattern screener and report results."""
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text(
        "📊 <b>Tiingo scan running…</b> chart patterns + quality gates (~30s)",
        parse_mode="HTML"
    )

    def _run():
        from agents.tiingo_screener import screen
        from data.custom_watchlist import CUSTOM_WATCHLIST
        try:
            from data.universe import get_nasdaq_watchlist, get_dow_watchlist
            watchlist = list(dict.fromkeys(
                CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
            ))
        except Exception:
            watchlist = CUSTOM_WATCHLIST
        return screen(watchlist)

    try:
        results = await _run_in_thread(_run)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Tiingo scan failed: {e}")
        return

    if not results:
        await update.message.reply_text("No signals above WATCH threshold today.")
        return

    tier_emoji = {"STRONG BUY": "🟢", "BUY": "🔵", "WATCH": "🟡", "SKIP": "⚪"}
    strong = [r for r in results if r.signal_tier == "STRONG BUY"]
    buys   = [r for r in results if r.signal_tier == "BUY"]
    watch  = [r for r in results if r.signal_tier == "WATCH"]

    lines = [f"<b>📊 Tiingo Chart Screen — {len(results)} signals</b>\n"]

    def _fmt_row(r):
        em = tier_emoji.get(r.signal_tier, "⚪")
        return (
            f"{em} <b>{r.ticker}</b>  {r.score:.0f}pts  "
            f"{r.pattern}  [{r.state}]  "
            f"vol={r.rel_vol:.1f}x  ATR={r.atr_ratio:.2f}x"
        )

    if strong:
        lines.append("🟢 <b>STRONG BUY</b>")
        lines.extend(_fmt_row(r) for r in strong)
    if buys:
        lines.append("\n🔵 <b>BUY</b>")
        lines.extend(_fmt_row(r) for r in buys[:8])
    if watch:
        lines.append(f"\n🟡 <b>WATCH</b> ({len(watch)} signals)")
        lines.extend(_fmt_row(r) for r in watch[:5])

    msg = "\n".join(lines)
    try:
        await update.message.reply_text(msg[:4096], parse_mode="HTML")
    except Exception:
        plain = msg.replace("<b>","").replace("</b>","")
        await update.message.reply_text(plain[:4096])


async def cmd_ib(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the Institutional Breakout scanner on demand."""
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text(
        "📈 <b>Institutional Breakout scan running…</b> Setup B (~45s)",
        parse_mode="HTML"
    )

    def _run():
        from agents.signal_router import route_signals
        from agents.market_regime import compute_regime
        from data.custom_watchlist import CUSTOM_WATCHLIST
        try:
            from data.universe import get_nasdaq_watchlist, get_dow_watchlist
            watchlist = list(dict.fromkeys(
                CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=50) + get_dow_watchlist()
            ))
        except Exception:
            watchlist = CUSTOM_WATCHLIST
        regime = compute_regime()
        return route_signals(watchlist, regime=regime, run_momentum=False, run_ib=True)

    try:
        signals = await _run_in_thread(_run)
    except Exception as e:
        await update.message.reply_text(f"⚠️ IB scan failed: {e}")
        return

    if not signals:
        await update.message.reply_text("No Setup B signals today.")
        return

    lines = [f"<b>📈 Institutional Breakout — {len(signals)} signals</b>\n"]
    for s in signals[:10]:
        action = "🟢" if s["strategy"] == "institutional_breakout" else "🔵"
        lines.append(
            f"{action} <b>{s['ticker']}</b>  {s['score']:.0f}pts  "
            f"[{s['strategy'].replace('_',' ')}]  "
            f"entry={s['entry']:.2f}  stop={s['stop'] or 'regime'}  "
            f"ATR={s['atr'] or '—'}"
        )
    try:
        await update.message.reply_text("\n".join(lines)[:4096], parse_mode="HTML")
    except Exception:
        plain = "\n".join(lines).replace("<b>","").replace("</b>","")
        await update.message.reply_text(plain[:4096])


async def cmd_strategy_perf(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show per-strategy performance attribution from the trade journal."""
    if not _authorised(update):
        return await _deny(update)

    def _run():
        from utils.trade_journal import performance_by_strategy
        return performance_by_strategy()

    try:
        perf = await _run_in_thread(_run)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed: {e}")
        return

    if not perf:
        await update.message.reply_text("No closed trades in journal yet.")
        return

    lines = ["<b>📊 Strategy Attribution</b>\n"]
    for strat, d in sorted(perf.items()):
        flag = " ⚠️ low-n" if d["low_n"] else ""
        pf   = f"{d['profit_factor']:.2f}" if d["profit_factor"] else "—"
        lines.append(
            f"<b>{strat}</b>{flag}\n"
            f"  Trades: {d['trades']} | Win: {d['win_rate']:.0%} | "
            f"Avg ret: {d['avg_return']:+.2%} | PF: {pf}"
        )
    await update.message.reply_text("\n".join(lines)[:4096], parse_mode="HTML")


async def cmd_research(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return await _deny(update)
    await update.message.reply_text("🔬 Running autoresearch analysis…")

    def _run():
        try:
            from utils.autoresearch import win_rate_by_indicator, _baseline_win_rate, \
                STRONG_EDGE_THRESHOLD, WEAK_EDGE_THRESHOLD
            from utils.trade_journal import get_closed_trades
            scores   = win_rate_by_indicator()
            baseline = _baseline_win_rate()
            trades   = get_closed_trades(min_trades=0)
            n        = len([t for t in trades if t["profitable"] is not None])

            lines = [f"<b>🔬 Autoresearch</b>  ({n} closed trades, baseline {baseline:.0%})\n"]

            if not scores:
                lines.append("Not enough data yet — need ≥10 trades per indicator condition.")
            else:
                strong = [(k, v) for k, v in scores.items()
                          if v["win_rate"] - baseline >= STRONG_EDGE_THRESHOLD]
                weak   = [(k, v) for k, v in scores.items()
                          if v["win_rate"] - baseline <= WEAK_EDGE_THRESHOLD]
                normal = [(k, v) for k, v in scores.items()
                          if k not in dict(strong) and k not in dict(weak)]

                if strong:
                    lines.append("🟢 <b>Strong signals</b>")
                    for k, v in sorted(strong, key=lambda x: -x[1]["win_rate"]):
                        lines.append(f"  {k}: {v['win_rate']:.0%} (n={v['n']})")

                if weak:
                    lines.append("\n🔴 <b>Weak signals (tiebreaker only)</b>")
                    for k, v in sorted(weak, key=lambda x: x[1]["win_rate"]):
                        lines.append(f"  {k}: {v['win_rate']:.0%} (n={v['n']})")

                lines.append(f"\n⚪ {len(normal)} signals at baseline weight")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ Autoresearch error: {e}"

    result = await _run_in_thread(_run)
    await update.message.reply_text(result, parse_mode="HTML")


# ── Single-instance lock ───────────────────────────────────────────────────────
# Uses a Windows named mutex (primary) so the OS releases the lock automatically
# when the process dies — even on forceful kill.  No PID-reuse false-positives.

import atexit, pathlib

_LOCK_FILE = pathlib.Path(__file__).parent / "logs" / "telegram_bot.pid"
_MUTEX_HANDLE = None   # kept alive so GC doesn't close it

def _acquire_lock() -> bool:
    """Return True if we obtained the single-instance lock, False if already held."""
    global _MUTEX_HANDLE

    # ── Primary: Windows named mutex ─────────────────────────────────────────
    # Use the GLOBAL namespace (not Local\) so the lock is machine-wide across
    # ALL sessions. A Local\ mutex is per-login-session, so a process started by
    # the watchdog in one session and one started elsewhere would NOT see each
    # other's lock -- both would acquire, both would poll Telegram, and collide
    # with 409 Conflict (crash-loop). Global\ makes the single-instance guard
    # actually single-instance across the whole machine.
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        _MUTEX_NAME = "Global\\TelegramTradingBot_SingleInstance"
        # bInitialOwner=False: we don't need ownership, only existence detection.
        # Create with default (non-inheritable) handle so a child process spawned
        # by this one can NOT inherit the handle and falsely "own" the lock.
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        handle = k32.CreateMutexW(None, False, _MUTEX_NAME)
        err    = ctypes.get_last_error() or k32.GetLastError()
        # ERROR_ALREADY_EXISTS (183) means another LIVE process holds this named
        # mutex -> we are a duplicate, refuse to start. This is the definitive
        # cross-session, cross-interpreter single-instance check.
        if err == 183:
            if handle:
                k32.CloseHandle(handle)
            logger.warning("Telegram lock held by another instance (global mutex) — exiting.")
            return False
        if handle:
            _MUTEX_HANDLE = handle          # keep reference — released on process exit
            _LOCK_FILE.parent.mkdir(exist_ok=True)
            _LOCK_FILE.write_text(str(os.getpid()))
            atexit.register(_release_lock)
            return True
        logger.debug("Mutex handle was null, falling back to PID file.")
    except Exception as _mutex_err:
        logger.debug(f"Mutex lock failed ({_mutex_err}), falling back to PID file.")

    # ── Fallback: PID-file with cmdline verification ──────────────────────────
    _LOCK_FILE.parent.mkdir(exist_ok=True)
    if _LOCK_FILE.exists():
        try:
            old_pid = int(_LOCK_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                try:
                    proc    = psutil.Process(old_pid)
                    cmdline = " ".join(proc.cmdline())
                    if "telegram_bot" in cmdline:
                        logger.warning(f"Telegram lock held by PID {old_pid} (PID file).")
                        return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
    _LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(_release_lock)
    return True

def _release_lock() -> None:
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("\nERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("  1. Message @BotFather on Telegram → /newbot")
        print("  2. Copy the token and add to .env:  TELEGRAM_BOT_TOKEN=<token>")
        print("  3. Restart this script\n")
        return

    # ── Prevent duplicate instances ───────────────────────────────────────────
    # NOTE: do NOT auto-pip-install psutil here. The previous version ran
    # subprocess.run([sys.executable, "-m", "pip", ...]) on startup, and under a
    # uv-based venv sys.executable resolves to the uv BASE python — which spawned
    # a SECOND telegram process on a different interpreter that lacked psutil,
    # producing two pollers fighting the same bot token (409 crash-loop). The
    # mutex lock below doesn't need psutil (only the PID-file fallback does, and
    # that's optional), so we simply skip if it's missing.
    try:
        import psutil  # noqa: F401 — used only by the PID-file fallback
    except ImportError:
        logger.debug("psutil not installed — PID-file fallback disabled (mutex lock still works).")

    if not _acquire_lock():
        print("\n⚠️  Another telegram_bot.py is already running.")
        print("   Close the other window first, or run:")
        print("   Get-Process python | Stop-Process -Force\n")
        return

    def _build_app():
        return (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .read_timeout(30)
            .write_timeout(30)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
        )

    def _register_handlers(a):
        a.add_handler(CommandHandler("start",     cmd_start))
        a.add_handler(CommandHandler("help",      cmd_help))
        a.add_handler(CommandHandler("account",   cmd_account))
        a.add_handler(CommandHandler("positions", cmd_positions))
        a.add_handler(CommandHandler("options",   cmd_options))
        a.add_handler(CommandHandler("scan",      cmd_scan))
        a.add_handler(CommandHandler("nasdaq",    cmd_nasdaq))
        a.add_handler(CommandHandler("dow",       cmd_dow))
        a.add_handler(CommandHandler("add",        cmd_add))
        a.add_handler(CommandHandler("remove",     cmd_remove))
        a.add_handler(CommandHandler("watchlist",     cmd_watchlist))
        a.add_handler(CommandHandler("tiingo",        cmd_tiingo))
        a.add_handler(CommandHandler("ib",            cmd_ib))
        a.add_handler(CommandHandler("perf",          cmd_strategy_perf))
        a.add_handler(CommandHandler("research",      cmd_research))
        return a

    app = _register_handlers(_build_app())

    from config import ENV, IS_PROD, PAPER_TRADING
    _mode = "LIVE" if (IS_PROD and not PAPER_TRADING) else "PAPER"
    print(f"\n✅ Telegram bot running  env={ENV.upper()}  mode={_mode}")
    print("   Press Ctrl+C to stop.\n")
    logger.info(f"Telegram bot started  env={ENV}  mode={_mode}")
    from utils.version import build_stamp as _build_stamp
    logger.info(f"BUILD | {_build_stamp()}")
    _health.mark_ready()

    # Auto-reconnect on network errors
    import time as _time
    while True:
        try:
            # drop_pending_updates=False: do NOT discard queued commands on
            # (re)start. With the bot restarting during today's churn, =True was
            # wiping /scan before it could be handled. False lets queued commands
            # process after a restart.
            app.run_polling(drop_pending_updates=False)
            # run_polling returned cleanly (shouldn't happen in normal operation)
            logger.warning("run_polling returned without exception — restarting in 5s...")
            _time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Telegram bot stopped by user.")
            break
        except SystemExit as e:
            # PTB sometimes raises SystemExit on fatal errors; treat as reconnectable
            logger.warning(f"Telegram run_polling raised SystemExit({e.code}) — reconnecting in 10s...")
            _time.sleep(10)
            app = _register_handlers(_build_app())
        except Exception as e:
            logger.warning(f"Telegram connection lost: {e} — reconnecting in 10s...")
            _time.sleep(10)
            app = _register_handlers(_build_app())
            logger.info("Reconnecting Telegram bot...")


if __name__ == "__main__":
    main()
