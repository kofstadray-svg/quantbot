"""
utils/autoresearch.py — Lightweight Karpathy-style prompt autoresearch.

HOW IT WORKS:
  1. Reads trade_journal.db — win rates per indicator condition
  2. Compares each indicator's win rate to the baseline (overall win rate)
  3. Generates a modified SYSTEM_PROMPT that upweights proven signals
     and downweights weak/unreliable ones
  4. Saves candidate prompt to prompts/candidate.txt
  5. After 5 trading days, compare_and_commit() checks if candidate Sharpe
     is better than baseline — keep or revert

USAGE:
  python utils/autoresearch.py          # generate candidate prompt
  python utils/autoresearch.py --status # show current indicator scores
  python utils/autoresearch.py --commit # commit candidate if better, else revert

The loop runs itself — wire it into a weekly cron or run manually on Mondays.
"""
from __future__ import annotations
import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from loguru import logger

from utils.trade_journal import win_rate_by_indicator, get_closed_trades

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
CANDIDATE   = PROMPTS_DIR / "candidate.txt"
BASELINE    = PROMPTS_DIR / "baseline.txt"
SCORES_LOG  = PROMPTS_DIR / "scores_history.jsonl"

# How much a win-rate deviation from baseline must be before we change weighting
STRONG_EDGE_THRESHOLD  = 0.12   # win_rate >= baseline + 12% → STRONG signal
WEAK_EDGE_THRESHOLD    = -0.10  # win_rate <= baseline - 10% → demote to tiebreaker


def _ensure_dirs() -> None:
    PROMPTS_DIR.mkdir(exist_ok=True)


def _baseline_win_rate() -> float:
    trades = get_closed_trades(min_trades=0)
    if not trades:
        return 0.50
    closed = [t for t in trades if t["profitable"] is not None]
    if not closed:
        return 0.50
    return sum(t["profitable"] for t in closed) / len(closed)


def _sharpe(trades: list[dict], window_days: int = 30) -> float | None:
    """Rolling Sharpe over the last window_days of closed trades."""
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    recent = [t for t in trades if t.get("exit_date", "") >= cutoff
              and t["return_pct"] is not None]
    if len(recent) < 5:
        return None
    returns = [t["return_pct"] for t in recent]
    mean = sum(returns) / len(returns)
    std  = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
    return (mean / std * math.sqrt(252)) if std > 0 else 0.0


def generate_candidate_prompt() -> str:
    """
    Read indicator win rates and produce a modified SYSTEM_PROMPT.
    Returns the candidate prompt string.
    """
    _ensure_dirs()

    scores = win_rate_by_indicator()
    baseline_wr = _baseline_win_rate()
    trades      = get_closed_trades(min_trades=0)
    n_trades    = len([t for t in trades if t["profitable"] is not None])

    logger.info(f"Autoresearch | baseline win rate: {baseline_wr:.1%}  n={n_trades}")

    # Categorise each signal
    strong:    list[str] = []
    normal:    list[str] = []
    tiebreaker: list[str] = []

    label_map = {
        "golden_cross_true":  "golden_cross=true (50MA above 200MA)",
        "golden_cross_false": "golden_cross=false",
        "obv_rising":         "obv_trend='rising'",
        "obv_falling":        "obv_trend='falling'",
        "obv_flat":           "obv_trend='flat'",
        "rsi_oversold":       "rsi_14 < 35 (oversold)",
        "rsi_healthy":        "rsi_14 35-50 (healthy momentum)",
        "rsi_elevated":       "rsi_14 50-70 (elevated)",
        "rsi_overbought":     "rsi_14 > 70 (overbought)",
        "macd_bullish":       "macd_signal > 0 (bullish histogram)",
        "macd_bearish":       "macd_signal < 0 (bearish histogram)",
        "stoch_oversold":     "stoch_k < 30 (oversold)",
        "stoch_overbought":   "stoch_k > 70 (overbought)",
        "stoch_neutral":      "stoch_k 30-70 (neutral)",
        "vol_rsi_bull":       "volume_rsi > 55 (bull volume dominant)",
        "vol_rsi_bear":       "volume_rsi < 55 (bear volume dominant)",
        "bb_near_lower":      "bb_position < 0.2 (near lower band)",
        "bb_near_upper":      "bb_position > 0.8 (near upper band)",
        "bb_middle":          "bb_position 0.2-0.8 (mid-range)",
    }

    score_detail: list[dict] = []

    for key, stats in scores.items():
        wr    = stats["win_rate"]
        n     = stats["n"]
        delta = wr - baseline_wr
        label = label_map.get(key, key)
        score_detail.append({"signal": key, "win_rate": wr, "n": n, "delta": delta})

        if delta >= STRONG_EDGE_THRESHOLD:
            strong.append(f"    - {label}  [win rate {wr:.0%}, n={n}]")
        elif delta <= WEAK_EDGE_THRESHOLD:
            tiebreaker.append(f"    - {label}  [weak: win rate {wr:.0%}, n={n}]")
        else:
            normal.append(f"    - {label}")

    # Log scores for trend tracking
    SCORES_LOG.parent.mkdir(exist_ok=True)
    with open(SCORES_LOG, "a") as f:
        f.write(json.dumps({
            "date": date.today().isoformat(),
            "baseline_win_rate": baseline_wr,
            "n_trades": n_trades,
            "scores": score_detail,
        }) + "\n")

    # Build prompt sections
    strong_section = (
        "\n  HISTORICALLY STRONG SIGNALS — weight these most heavily (+2 each):\n"
        + "\n".join(strong)
    ) if strong else ""

    tiebreaker_section = (
        "\n  WEAK SIGNALS — treat as tiebreakers only (+0.5 each, not enough alone):\n"
        + "\n".join(tiebreaker)
    ) if tiebreaker else ""

    normal_section = (
        "\n  STANDARD SIGNALS — score as normal (+1 each):\n"
        + "\n".join(normal)
    ) if normal else ""

    # Read the baseline prompt and inject the learned weighting block
    base_prompt = _load_baseline_prompt()

    learned_block = f"""
LEARNED SIGNAL WEIGHTS (auto-updated from {n_trades} closed trades, baseline win rate {baseline_wr:.0%}):
{strong_section}
{normal_section}
{tiebreaker_section}

Apply these weights INSTEAD OF the static scoring rules above when they conflict.
""" if (strong or tiebreaker) else ""

    # Insert learned block before the BUY/HOLD/SKIP criteria line
    if "BUY criteria" in base_prompt and learned_block:
        candidate = base_prompt.replace(
            "BUY criteria",
            learned_block + "\nBUY criteria",
            1,
        )
    else:
        candidate = base_prompt + learned_block

    CANDIDATE.write_text(candidate, encoding="utf-8")
    logger.info(
        f"Autoresearch | candidate prompt written to {CANDIDATE}  "
        f"(strong={len(strong)} tiebreaker={len(tiebreaker)})"
    )
    return candidate


def _load_baseline_prompt() -> str:
    """Return the baseline prompt — falls back to the static one in stock_screener."""
    if BASELINE.exists():
        return BASELINE.read_text(encoding="utf-8")
    # First run: save the trend-following prompt as the baseline (regime=None
    # returns TREND_FOLLOWING_PROMPT, the original single-prompt baseline).
    from agents.stock_screener import get_system_prompt
    baseline = get_system_prompt(regime=None)
    BASELINE.write_text(baseline, encoding="utf-8")
    logger.info(f"Autoresearch | baseline prompt saved to {BASELINE}")
    return baseline


def load_active_prompt() -> str:
    """
    Return the best available prompt for use in stock_screener.
    If a validated candidate exists, use it; otherwise fall back to baseline.
    Called by stock_screener._compute_indicators at runtime.
    """
    active = PROMPTS_DIR / "active.txt"
    if active.exists():
        return active.read_text(encoding="utf-8")
    return _load_baseline_prompt()


def compare_and_commit(trial_days: int = 5) -> None:
    """
    Compare Sharpe of candidate vs baseline over the last trial_days.
    Keep candidate (write to active.txt) if better, else revert.
    """
    _ensure_dirs()
    trades = get_closed_trades(min_trades=0)

    baseline_sharpe = _sharpe(trades, window_days=trial_days * 2)  # prior period
    candidate_sharpe = _sharpe(trades, window_days=trial_days)      # recent period

    logger.info(
        f"Autoresearch | compare: "
        f"baseline_sharpe={baseline_sharpe}  candidate_sharpe={candidate_sharpe}"
    )

    active = PROMPTS_DIR / "active.txt"

    if baseline_sharpe is None or candidate_sharpe is None:
        logger.warning("Autoresearch | not enough data to compare — keeping current.")
        return

    if candidate_sharpe > baseline_sharpe:
        if CANDIDATE.exists():
            active.write_text(CANDIDATE.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(
            f"✅ Autoresearch COMMIT: candidate Sharpe {candidate_sharpe:.2f} "
            f"> baseline {baseline_sharpe:.2f} — prompt promoted to active."
        )
    else:
        if BASELINE.exists():
            active.write_text(BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(
            f"⏪ Autoresearch REVERT: candidate Sharpe {candidate_sharpe:.2f} "
            f"<= baseline {baseline_sharpe:.2f} — reverting to baseline."
        )


def print_status() -> None:
    """Print a human-readable indicator scorecard to stdout."""
    scores = win_rate_by_indicator()
    baseline = _baseline_win_rate()
    trades = get_closed_trades(min_trades=0)
    n = len([t for t in trades if t["profitable"] is not None])

    print(f"\n{'='*58}")
    print(f"  AUTORESEARCH STATUS  |  {n} closed trades  |  baseline {baseline:.0%}")
    print(f"{'='*58}")
    if not scores:
        print("  Not enough data yet (need ≥10 trades per indicator).")
    else:
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1]["win_rate"])
        for key, stats in sorted_scores:
            delta = stats["win_rate"] - baseline
            bar = "🟢" if delta >= STRONG_EDGE_THRESHOLD else (
                  "🔴" if delta <= WEAK_EDGE_THRESHOLD else "⚪")
            print(
                f"  {bar} {key:<30}  {stats['win_rate']:.0%}  "
                f"({delta:+.0%})  n={stats['n']}"
            )
    print(f"{'='*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.commit:
        compare_and_commit()
    else:
        generate_candidate_prompt()
        print_status()
