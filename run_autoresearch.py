"""
run_autoresearch.py — Weekly autoresearch loop runner.

Run every Monday morning before the market opens:
  python run_autoresearch.py          # generate candidate + status report
  python run_autoresearch.py --commit # compare and keep/revert after 5-day trial

WHAT HAPPENS:
  1. Reads trade_journal.db — all closed trades since bot launch
  2. Scores each indicator condition by win rate vs baseline
  3. Writes a modified SYSTEM_PROMPT to prompts/candidate.txt
     - Proven signals (+12%+ above baseline win rate) → STRONG (+2 pts each)
     - Weak signals  (-10%+ below baseline win rate)  → tiebreaker only (+0.5 pts)
  4. stock_screener.py automatically loads prompts/active.txt at runtime
  5. After 5 trading days, run with --commit to lock in or revert

HOW TO SCHEDULE (Windows Task Scheduler):
  Action: python run_autoresearch.py
  Trigger: Every Monday at 8:45 AM

  Action: python run_autoresearch.py --commit
  Trigger: Every Monday at 8:50 AM  (after 1 week of trial)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from utils.autoresearch import generate_candidate_prompt, compare_and_commit, print_status
from utils.trade_journal import get_closed_trades

from utils.logging import setup_logging
setup_logging("autoresearch", rotation="1 week", retention="12 weeks")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-style prompt autoresearch")
    parser.add_argument("--commit", action="store_true",
                        help="Compare candidate vs baseline Sharpe and keep/revert")
    parser.add_argument("--status", action="store_true",
                        help="Print indicator scorecard only")
    args = parser.parse_args()

    trades = get_closed_trades(min_trades=0)
    n_closed = len([t for t in trades if t["profitable"] is not None])

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH  |  {n_closed} closed trades in journal")
    print(f"{'='*60}\n")

    if args.status:
        print_status()
        return

    if args.commit:
        compare_and_commit(trial_days=5)
        print_status()
        return

    # Default: generate candidate prompt
    if n_closed < 20:
        print(f"  ⚠️  Only {n_closed} closed trades — need ≥20 for reliable signal scoring.")
        print("  Showing current status. No candidate generated yet.\n")
        print_status()
        return

    candidate = generate_candidate_prompt()
    print_status()

    print("\n  📝 Candidate prompt written to prompts/candidate.txt")
    print("  The bot will continue using prompts/active.txt (unchanged).")
    print("  After 5 trading days, run:  python run_autoresearch.py --commit")
    print("  to lock in the new prompt if Sharpe improved, or revert if not.\n")


if __name__ == "__main__":
    main()
