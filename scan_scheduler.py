"""
scan_scheduler.py — Runs morning_scan.py automatically at:
  10:30 AM  (1 hour after market open)
   2:30 PM  (1 hour before market close)
  Monday – Friday only.

After each scan, a summary is sent to your Telegram bot.

HOW TO RUN:
  python scan_scheduler.py

Keep this running in the background (or use the desktop shortcut).
Press Ctrl+C to stop.
"""
import schedule
import time
import subprocess
import sys
import os
import re
import requests
from datetime import datetime
from loguru import logger

from utils.logging import setup_logging
setup_logging("scheduler")

SCRIPT      = os.path.join(os.path.dirname(__file__), "morning_scan.py")
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(SCRIPT_DIR, "logs")


# ── Telegram helper ─────────────────────────────────────────────────────────────

def _send_telegram(message: str) -> None:
    """Send a message to the configured Telegram chat."""
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured — skipping notification.")
            return
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram notification sent.")
        else:
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram notification error: {e}")


# ── Log parser ──────────────────────────────────────────────────────────────────

def _parse_log(label: str, elapsed_sec: float) -> str:
    """
    Read today's morning_scan log and build a Telegram summary message.
    """
    today     = datetime.now().strftime("%Y-%m-%d")
    log_path  = os.path.join(LOG_DIR, f"morning_scan_{today}.log")

    buys      = []   # (ticker, setups, passed, total)
    watches   = []   # ticker strings
    trades    = []   # ticker strings that were AUTO-TRADEd
    holds     = []   # BUY signals skipped for no entry setup
    skipped   = 0
    scanned   = 0

    if not os.path.exists(log_path):
        return f"📊 <b>{label}</b>\n\nLog file not found — scan may still be running."

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Only look at lines written during this scan run (last N lines after scan start)
    # Find the last "Morning scan started" marker
    start_idx = 0
    for i, line in enumerate(lines):
        if "Morning scan started" in line:
            start_idx = i
    lines = lines[start_idx:]

    for line in lines:
        # Count tickers scanned
        if "Momentum scan:" in line and "tickers" not in line:
            scanned += 1

        # BUY / WATCH / SKIP signals
        m = re.search(r"Momentum \| (\w+)\s+(BUY|WATCH|SKIP)\s+(\d+)/(\d+).*?Entry=\[([^\]]*)\]", line)
        if m:
            ticker, signal, passed, total, entry_raw = m.groups()
            entry_list = [e.strip() for e in entry_raw.split(",") if e.strip() and e.strip() != "none"]
            if signal == "BUY":
                buys.append((ticker, entry_list, int(passed), int(total)))
            elif signal == "WATCH":
                watches.append(ticker)
            elif signal == "SKIP":
                skipped += 1

        # Auto-trades placed
        if "AUTO-TRADE ->" in line:
            m2 = re.search(r"AUTO-TRADE -> (\w+)", line)
            if m2:
                trades.append(m2.group(1))

        # BUY signals held back (no entry setup)
        if "HOLD ->" in line and "NO entry setup" in line:
            m3 = re.search(r"HOLD -> (\w+)", line)
            if m3:
                holds.append(m3.group(1))

    # ── Format message ──────────────────────────────────────────────────────────
    mins = int(elapsed_sec // 60)
    secs = int(elapsed_sec % 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    now  = datetime.now().strftime("%I:%M %p")

    lines_out = [f"📊 <b>{label}</b>  [{now}]\n"]

    if buys:
        lines_out.append(f"🟢 <b>BUY Signals ({len(buys)})</b>")
        for ticker, setups, passed, total in buys:
            setup_str = ", ".join(setups) if setups else "no entry setup"
            trade_tag = " ✅ <i>traded</i>" if ticker in trades else " ⏸ <i>no setup</i>"
            lines_out.append(f"  • <b>{ticker}</b> — {setup_str} | {passed}/{total} checks{trade_tag}")
    else:
        lines_out.append("🔴 No BUY signals")

    if watches:
        lines_out.append(f"\n👁 <b>WATCH ({len(watches)})</b>: {', '.join(watches)}")

    if holds:
        lines_out.append(f"\n⏸ <b>Held (no setup)</b>: {', '.join(holds)}")

    if trades:
        lines_out.append(f"\n🤖 <b>Auto-trades placed</b>: {', '.join(trades)}")
    else:
        lines_out.append("\n🤖 No trades placed")

    lines_out.append(f"\n⏱ {scanned} tickers scanned in {time_str}")

    return "\n".join(lines_out)


# ── Scan runner ─────────────────────────────────────────────────────────────────

def run_scan(label: str) -> None:
    if datetime.now().weekday() >= 5:
        logger.info(f"{label}: skipping — weekend.")
        return

    logger.info(f"{'='*50}")
    logger.info(f"  {label} — starting morning_scan.py")
    logger.info(f"{'='*50}")
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  {label} — launching morning_scan.py...\n")

    # Notify Telegram that scan is starting
    _send_telegram(f"🔍 <b>{label} starting…</b>\nScanning Nasdaq + Dow + Watchlist + Chart Patterns")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, SCRIPT],
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(f"{label} completed in {elapsed:.0f}s.")
        else:
            logger.warning(f"{label} exited with code {result.returncode}.")
            _send_telegram(f"⚠️ <b>{label}</b>\nScan exited with error code {result.returncode}.")
            return

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"{label} failed to launch: {e}")
        _send_telegram(f"❌ <b>{label} FAILED</b>\n{e}")
        return

    # Parse log and send summary
    summary = _parse_log(label, elapsed)
    _send_telegram(summary)


# ── Schedule ────────────────────────────────────────────────────────────────────

schedule.every().day.at("10:30").do(run_scan, label="10:30 AM Scan")
schedule.every().day.at("14:30").do(run_scan, label="2:30 PM Scan")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*54)
    print("  SCAN SCHEDULER  +  TELEGRAM NOTIFICATIONS")
    print("  Weekday scans at:")
    print("    10:30 AM  — 1 hour after market open")
    print("     2:30 PM  — 1 hour before market close")
    print("  Results sent to Telegram after each scan.")
    print("  Press Ctrl+C to stop.")
    print("="*54 + "\n")

    logger.info("Scan scheduler started.")

    next_run = schedule.next_run()
    if next_run:
        print(f"  Next scan: {next_run.strftime('%A %I:%M %p')}\n")

    # Send startup ping to Telegram
    _send_telegram(
        "✅ <b>Scan Scheduler Online</b>\n\n"
        "Scans will run at:\n"
        "  • 10:30 AM (1hr after open)\n"
        "  • 2:30 PM (1hr before close)\n\n"
        "Results will be sent here automatically."
    )

    while True:
        schedule.run_pending()
        time.sleep(20)
