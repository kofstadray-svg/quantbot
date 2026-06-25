"""
backtesting/monte_carlo.py — Monte Carlo simulation with block bootstrap.

Two sampling methods:

  run()              : IID resample (original)
                       Treats every trade as independent. Fast, simple.
                       Overestimates confidence when trades cluster (trend regimes).

  run_block()        : Block bootstrap (NEW)
                       Resamples consecutive blocks of N trades to preserve the
                       serial correlation that exists in real trade sequences —
                       e.g. a string of winners in a bull market, then a string
                       of losers in a correction. This gives wider, more honest
                       confidence intervals.

  run_regime_mc()    : Per-regime Monte Carlo (NEW)
                       Runs IID Monte Carlo separately within each regime bucket
                       (bull / bear / high_vol / extreme_vol) so you can see
                       whether the strategy is actually profitable in bad markets
                       or just riding bull-market beta.

Rule of thumb:
  Use run_block() as your primary validation — it is more conservative.
  If run_block() 5th-percentile > 0, the strategy is genuinely robust.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class SimResult:
    median_return:         float
    percentile_5:          float
    percentile_95:         float
    prob_ruin:             float    # fraction of runs that hit ruin_threshold
    expected_final_equity: float
    runs:                  int
    method:                str = "iid"   # "iid" or "block"


# ---------------------------------------------------------------------------
# IID resample (original method — preserved for backward compat)
# ---------------------------------------------------------------------------

def run(
    trade_returns: list[float],
    starting_equity: float = 1000.0,
    n_runs: int = 1000,
    n_periods: int | None = None,
    ruin_threshold: float = 0.1,
    annual_trades: int = 252,
) -> SimResult:
    """
    Monte Carlo by sampling individual trades with replacement (IID).

    trade_returns   : per-trade P&L as fractions (e.g. 0.05 = +5%)
    starting_equity : portfolio start value
    n_runs          : number of simulated paths
    n_periods       : trades per path; defaults to min(len, annual_trades)
    ruin_threshold  : equity fraction below which run is counted as "ruined"
    annual_trades   : cap on n_periods to avoid multi-year compounding
    """
    if not trade_returns:
        raise ValueError("trade_returns is empty — nothing to simulate.")

    returns = np.clip(np.array(trade_returns, dtype=float), -0.50, 0.50)
    if n_periods is None:
        n_periods = min(len(returns), annual_trades)

    rng = np.random.default_rng(seed=42)
    final_equities = np.empty(n_runs)
    ruined = 0

    for i in range(n_runs):
        sampled      = rng.choice(returns, size=n_periods, replace=True)
        equity_curve = starting_equity * np.cumprod(1 + sampled)
        if np.any(equity_curve < starting_equity * ruin_threshold):
            ruined += 1
        final_equities[i] = equity_curve[-1]

    result = SimResult(
        median_return         = (np.median(final_equities) - starting_equity) / starting_equity,
        percentile_5          = (np.percentile(final_equities, 5)  - starting_equity) / starting_equity,
        percentile_95         = (np.percentile(final_equities, 95) - starting_equity) / starting_equity,
        prob_ruin             = ruined / n_runs,
        expected_final_equity = float(np.mean(final_equities)),
        runs                  = n_runs,
        method                = "iid",
    )
    logger.info(
        f"MC/IID ({n_runs} runs, {n_periods} trades):  "
        f"median={result.median_return:+.1%}  5th={result.percentile_5:+.1%}  "
        f"95th={result.percentile_95:+.1%}  ruin={result.prob_ruin:.1%}"
    )
    return result


# ---------------------------------------------------------------------------
# Block bootstrap — preserves serial correlation
# ---------------------------------------------------------------------------

def run_block(
    trade_returns: list[float],
    starting_equity: float = 1000.0,
    n_runs: int = 1000,
    block_size: int = 8,
    n_periods: int | None = None,
    ruin_threshold: float = 0.1,
    annual_trades: int = 252,
    seed: int = 42,
) -> SimResult:
    """
    Block bootstrap Monte Carlo.

    Instead of sampling individual trades, sample consecutive *blocks* of
    `block_size` trades. This preserves the serial dependence that exists in
    real trading: winning streaks during trending markets, losing streaks
    during drawdowns. The result is wider, more honest confidence intervals.

    block_size  : number of consecutive trades per block (default 8 ≈ 2 weeks)
                  Larger block = more correlation preserved, wider intervals.
                  Typical range: 5–20.
    """
    if not trade_returns:
        raise ValueError("trade_returns is empty.")

    returns = np.clip(np.array(trade_returns, dtype=float), -0.50, 0.50)
    n       = len(returns)

    if n_periods is None:
        n_periods = min(n, annual_trades)

    # Build all possible blocks (circular wrap so every trade can be a block start)
    blocks: list[np.ndarray] = []
    for start in range(n):
        block = np.array([returns[(start + j) % n] for j in range(block_size)])
        blocks.append(block)
    blocks_arr = np.array(blocks)  # shape: (n, block_size)

    n_blocks_needed = math.ceil(n_periods / block_size)

    rng = np.random.default_rng(seed=seed)
    final_equities = np.empty(n_runs)
    ruined = 0

    for i in range(n_runs):
        # Sample blocks, concatenate, trim to n_periods
        chosen_idx = rng.integers(0, len(blocks_arr), size=n_blocks_needed)
        sequence   = np.concatenate(blocks_arr[chosen_idx])[:n_periods]

        equity_curve = starting_equity * np.cumprod(1 + sequence)
        if np.any(equity_curve < starting_equity * ruin_threshold):
            ruined += 1
        final_equities[i] = equity_curve[-1]

    result = SimResult(
        median_return         = (np.median(final_equities) - starting_equity) / starting_equity,
        percentile_5          = (np.percentile(final_equities, 5)  - starting_equity) / starting_equity,
        percentile_95         = (np.percentile(final_equities, 95) - starting_equity) / starting_equity,
        prob_ruin             = ruined / n_runs,
        expected_final_equity = float(np.mean(final_equities)),
        runs                  = n_runs,
        method                = f"block(size={block_size})",
    )
    logger.info(
        f"MC/Block ({n_runs} runs, {n_periods} trades, block={block_size}):  "
        f"median={result.median_return:+.1%}  5th={result.percentile_5:+.1%}  "
        f"95th={result.percentile_95:+.1%}  ruin={result.prob_ruin:.1%}"
    )
    return result


# ---------------------------------------------------------------------------
# Per-regime Monte Carlo
# ---------------------------------------------------------------------------

def run_regime_mc(
    trades_by_regime: dict[str, list[float]],
    starting_equity: float = 1000.0,
    n_runs: int = 1000,
) -> dict[str, SimResult]:
    """
    Run a separate block bootstrap for each market regime.

    Args:
        trades_by_regime : {regime_label: [pnl_pct, ...]}
        starting_equity  : start equity for each regime sim

    Returns:
        {regime_label: SimResult}
    """
    results: dict[str, SimResult] = {}
    for regime, rets in trades_by_regime.items():
        if len(rets) < 10:
            continue
        try:
            results[regime] = run_block(
                rets,
                starting_equity=starting_equity,
                n_runs=n_runs,
            )
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(result: SimResult, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Monte Carlo Summary  ({result.runs} runs · {result.method}){tag}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Median return    : {result.median_return:+.1%}
 5th percentile   : {result.percentile_5:+.1%}  ← worst realistic outcome
 95th percentile  : {result.percentile_95:+.1%}  ← best realistic outcome
 Prob of ruin     : {result.prob_ruin:.1%}
 Avg final equity : ${result.expected_final_equity:,.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")


def print_regime_mc_summary(regime_results: dict[str, SimResult]) -> None:
    if not regime_results:
        return
    print("\n  PER-REGIME MONTE CARLO")
    print(f"  {'Regime':<14}  {'Runs':>5}  {'Median':>7}  {'5th':>7}  {'95th':>7}  {'Ruin':>5}")
    print(f"  {'─'*55}")
    order = ["bull", "neutral", "high_vol", "extreme_vol", "bear"]
    for regime in order + [r for r in regime_results if r not in order]:
        if regime not in regime_results:
            continue
        r = regime_results[regime]
        print(
            f"  {regime:<14}  {r.runs:>5}  "
            f"{r.median_return:>+6.1%}  {r.percentile_5:>+6.1%}  "
            f"{r.percentile_95:>+6.1%}  {r.prob_ruin:>4.0%}"
        )


if __name__ == "__main__":
    import math
    import random
    random.seed(1)
    # Demo: 55% win rate, 1.5:1 R:R
    demo_trades = [0.015 if random.random() < 0.55 else -0.01 for _ in range(150)]

    print("=== IID Resample ===")
    r1 = run(demo_trades, starting_equity=1000, n_runs=1000)
    print_summary(r1)

    print("\n=== Block Bootstrap (block=8) ===")
    r2 = run_block(demo_trades, starting_equity=1000, n_runs=1000, block_size=8)
    print_summary(r2)
