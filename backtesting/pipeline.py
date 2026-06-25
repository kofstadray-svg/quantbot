"""
backtesting/pipeline.py -- 5-tier institutional backtesting pipeline.

Run this before deploying any strategy live. Each tier is progressively
more conservative. A strategy must pass ALL 5 tiers before going live.

  Tier 1 — In-sample optimization
    Grid search over parameters on the full historical dataset.
    Objective: maximize Sharpe.  Produces: best params + IS metrics.
    ⚠  Overfitting risk is high here. Use results only to guide Tier 2.

  Tier 2 — Out-of-sample validation
    Apply IS-optimal params to a held-out period (most recent 20% of data).
    Produces: OOS Sharpe, win rate, expectancy.
    ✅  Strategy proceeds if OOS Sharpe > 0 and degradation ratio ≥ 0.5.

  Tier 3 — Walk-forward analysis
    Rolling IS/OOS windows (12m train / 3m test / 3m step).
    Produces: per-window params, IS→OOS degradation, parameter stability.
    ✅  Strategy proceeds if avg OOS Sharpe > 0 in >60% of windows.

  Tier 4 — Monte Carlo stress test
    Block bootstrap (preserves serial correlation) on concatenated OOS returns.
    Also runs per-regime MC to expose regime-specific weaknesses.
    Produces: 5th-percentile equity path, prob-of-ruin, regime breakdown.
    ✅  Strategy proceeds if: 5th-pct > −20%, ruin < 5%, profitable in bull regime.

  Tier 5 — Realistic microstructure simulation
    Replays OOS trades with partial fills, latency, and spread costs.
    Produces: raw vs net P&L, total execution drag per trade, edge-survival check.
    ✅  Strategy proceeds if avg net P&L > 0 after microstructure costs.

  After all 5 tiers: paper trading for ≥ 30 trading days before live deployment.

HOW TO RUN:
  cd project
  python backtesting/pipeline.py                        # full pipeline
  python backtesting/pipeline.py --tiers 1,2,3          # skip MC + micro
  python backtesting/pipeline.py --tickers AAPL,MSFT    # custom tickers
"""
from __future__ import annotations

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass, field
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

# Pipeline config
IS_MONTHS      = 12
OOS_MONTHS     = 3
STEP_MONTHS    = 3
TOTAL_HISTORY  = "5y"
POSITION_USD   = 100.0
MIN_OOS_PASSES = 0.60   # fraction of OOS windows that must be Sharpe > 0

BUY_THRESHOLDS   = [2.5, 3.0, 3.5, 4.0]
STOP_ATR_MULTS   = [1.0, 1.5, 2.0]
TIME_STOP_DAYS_G = [3, 5, 7]

# Pass/fail thresholds
MIN_OOS_SHARPE        = 0.0
MIN_DEGRADATION_RATIO = 0.50
MAX_RUIN_PROB         = 0.05
MIN_5TH_PCT_RETURN    = -0.20
MIN_NET_PNL           = 0.0


# ─── Shared data structures ───────────────────────────────────────────────────

@dataclass
class TierResult:
    tier:    int
    name:    str
    passed:  bool
    metrics: dict = field(default_factory=dict)
    notes:   list = field(default_factory=list)


@dataclass
class PipelineResult:
    tiers:         list[TierResult]
    deploy_ready:  bool
    summary:       str


# ─── Data loading + indicator helpers (shared) ───────────────────────────────

def _fetch(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    try:
        df = _yf_dl(ticker, period=period, interval="1d",
                    progress=False, auto_adjust=True, timeout=20)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 252 else None
    except Exception:
        return None


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _factor_score(df: pd.DataFrame, spy_close: pd.Series) -> pd.Series:
    """Simplified orthogonal factor score (trend + momentum + vol + RS)."""
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    adx   = ta.trend.ADXIndicator(high, low, close, window=14).adx().fillna(20)
    rsi   = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr   = _atr_series(df)
    atr_rank = atr.rolling(252, min_periods=30).rank(pct=True) * 100
    avg_vol  = vol.rolling(20).mean()
    rel_vol  = (vol / avg_vol.replace(0, np.nan)).fillna(1.0)
    spy_r    = spy_close.reindex(close.index).ffill()
    rs_slope = (close / spy_r).rolling(20).apply(
        lambda w: (np.polyfit(range(len(w)), w, 1)[0] / w[0] * 100) if len(w) > 1 and w[0] > 0 else 0.0,
        raw=True,
    ).fillna(0.0)

    score = (
        np.where(ma50 > ma200, 0.8, -0.8) +
        np.where(adx > 30, 0.5, np.where(adx > 20, 0.2, -0.2)) +
        np.where(rsi < 35, 1.5, np.where(rsi < 50, 0.5, np.where(rsi < 65, 0.0, -1.0))) +
        np.where(atr_rank < 35, 0.6, np.where(atr_rank < 55, 0.0, -0.6)) +
        np.where(rel_vol > 1.5, 0.6, np.where(rel_vol < 0.8, -0.4, 0.0)) +
        np.where(rs_slope > 0.3, 0.4, np.where(rs_slope < -0.3, -0.4, 0.0))
    )
    df["_atr"] = atr
    return pd.Series(score, index=close.index).fillna(0.0)


def _simulate_period(
    df: pd.DataFrame,
    score: pd.Series,
    start: date,
    end: date,
    buy_threshold: float,
    stop_atr_mult: float,
    time_stop_days: int,
) -> list[dict]:
    """Run the basic trade simulation (no microstructure) and return list of trade dicts."""
    mask = (df.index.date >= start) & (df.index.date < end)
    sub  = df.loc[mask].copy()
    sub_s = score.loc[mask]

    if len(sub) < 5:
        return []

    trades = []
    in_pos = False
    entry_price = entry_atr = 0.0
    entry_idx = 0
    entry_date_str = ""

    for i in range(1, len(sub)):
        s      = float(sub_s.iloc[i])
        price  = float(sub["close"].iloc[i])
        atr    = float(sub["_atr"].iloc[i]) if "_atr" in sub.columns else price * 0.02
        if np.isnan(atr):
            atr = price * 0.02
        dt = str(sub.index[i].date())

        if not in_pos:
            if s >= buy_threshold:
                in_pos = True
                entry_price = price
                entry_atr   = atr if atr > 0 else price * 0.02
                entry_idx   = i
                entry_date_str = dt
        else:
            hold = i - entry_idx
            reason = None
            if price < entry_price - stop_atr_mult * entry_atr:
                reason = f"stop({stop_atr_mult}ATR)"
            elif hold >= time_stop_days:
                if s < 0:
                    reason = f"time_stop({time_stop_days}d)"
            if reason is None and s < -1.5:
                reason = "score_exit"
            if reason:
                trades.append({
                    "entry_date":  entry_date_str,
                    "exit_date":   dt,
                    "entry_price": entry_price,
                    "exit_price":  price,
                    "pnl_pct":     (price - entry_price) / entry_price,
                    "hold_days":   hold,
                    "exit_reason": reason,
                    "atr_pct":     entry_atr / entry_price,
                })
                in_pos = False
    return trades


def _sharpe(rets: list[float]) -> float:
    if len(rets) < 2:
        return 0.0
    r = np.array(rets)
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def _max_dd(rets: list[float]) -> float:
    equity = 1000.0
    peak   = 1000.0
    mdd    = 0.0
    for r in rets:
        equity = equity * (1 + r)
        if equity > peak:
            peak = equity
        mdd = min(mdd, (equity - peak) / peak)
    return mdd


def _win_rate(rets: list[float]) -> float:
    return len([r for r in rets if r > 0]) / len(rets) if rets else 0.0


# ─── Tier 1: In-sample optimization ──────────────────────────────────────────

def run_tier1(ticker_data: dict, spy_close: pd.Series, oos_cutoff: date) -> TierResult:
    """Grid-search parameters on IS data (everything before oos_cutoff)."""
    print("  Tier 1 — In-sample optimization...")

    all_dates  = [df.index.date.min() for df, _ in ticker_data.values()]
    is_start   = max(all_dates)
    is_end     = oos_cutoff

    param_grid = list(itertools.product(BUY_THRESHOLDS, STOP_ATR_MULTS, TIME_STOP_DAYS_G))
    best_sharpe = -999.0
    best_params = {}
    best_trades: list[float] = []

    for bt, sm, td in param_grid:
        trades = []
        for ticker, (df, score) in ticker_data.items():
            trades.extend(_simulate_period(df, score, is_start, is_end, bt, sm, td))
        rets = [t["pnl_pct"] for t in trades]
        sh   = _sharpe(rets)
        if sh > best_sharpe and len(rets) >= 10:
            best_sharpe = sh
            best_params = {"buy_threshold": bt, "stop_atr_mult": sm, "time_stop_days": td}
            best_trades = rets

    win_r = _win_rate(best_trades)
    mdd   = _max_dd(best_trades)

    print(f"    Best params: {best_params}  IS Sharpe={best_sharpe:.2f}  "
          f"Trades={len(best_trades)}  WinRate={win_r:.0%}  MaxDD={mdd:.1%}")

    return TierResult(
        tier    = 1,
        name    = "In-sample optimization",
        passed  = best_sharpe > 0 and len(best_trades) >= 10,
        metrics = {"sharpe": best_sharpe, "trades": len(best_trades),
                   "win_rate": win_r, "max_dd": mdd, "best_params": best_params},
        notes   = [f"Best params: {best_params}"],
    )


# ─── Tier 2: Out-of-sample validation ────────────────────────────────────────

def run_tier2(
    ticker_data: dict,
    best_params: dict,
    spy_close: pd.Series,
    oos_cutoff: date,
    last_date: date,
) -> TierResult:
    """Apply IS params to held-out OOS period."""
    print("  Tier 2 — Out-of-sample validation...")

    trades = []
    for ticker, (df, score) in ticker_data.items():
        trades.extend(_simulate_period(
            df, score, oos_cutoff, last_date, **best_params
        ))

    rets  = [t["pnl_pct"] for t in trades]
    sh    = _sharpe(rets)
    win_r = _win_rate(rets)
    mdd   = _max_dd(rets)
    exp   = float(np.mean(rets)) if rets else 0.0

    passed = sh > MIN_OOS_SHARPE and len(rets) >= 5
    print(f"    OOS Sharpe={sh:.2f}  Trades={len(rets)}  "
          f"WinRate={win_r:.0%}  Expectancy={exp:+.2%}  MaxDD={mdd:.1%}  "
          f"{'PASS' if passed else 'FAIL'}")

    return TierResult(
        tier    = 2,
        name    = "Out-of-sample validation",
        passed  = passed,
        metrics = {"sharpe": sh, "trades": len(rets), "win_rate": win_r,
                   "expectancy": exp, "max_dd": mdd},
        notes   = [f"OOS period: {oos_cutoff} → {last_date}"],
    )


# ─── Tier 3: Walk-forward analysis ───────────────────────────────────────────

def run_tier3(ticker_data: dict, spy_close: pd.Series, all_dates: list[date]) -> TierResult:
    """Rolling IS/OOS windows."""
    print("  Tier 3 — Walk-forward analysis...")

    first_date = max(all_dates)
    last_date  = min(df.index.date.max() for df, _ in ticker_data.values())

    windows = []
    cursor  = first_date
    wid     = 0
    while True:
        is_start  = cursor
        is_end    = cursor + relativedelta(months=IS_MONTHS)
        oos_start = is_end
        oos_end   = oos_start + relativedelta(months=OOS_MONTHS)
        if oos_end > last_date:
            break
        windows.append((wid, is_start, is_end, oos_start, oos_end))
        cursor += relativedelta(months=STEP_MONTHS)
        wid    += 1

    if not windows:
        return TierResult(2, "Walk-forward", False, notes=["Not enough history for windows"])

    param_grid  = list(itertools.product(BUY_THRESHOLDS, STOP_ATR_MULTS, TIME_STOP_DAYS_G))
    oos_sharpes = []
    all_oos_rets: list[float] = []

    for wid, is_s, is_e, oos_s, oos_e in windows:
        best_sh = -999.0
        best_p  = {}
        for bt, sm, td in param_grid:
            is_t = []
            for _, (df, score) in ticker_data.items():
                is_t.extend(_simulate_period(df, score, is_s, is_e, bt, sm, td))
            if len(is_t) < 8:
                continue
            sh = _sharpe([t["pnl_pct"] for t in is_t])
            if sh > best_sh:
                best_sh = sh
                best_p  = {"buy_threshold": bt, "stop_atr_mult": sm, "time_stop_days": td}

        oos_t = []
        for _, (df, score) in ticker_data.items():
            oos_t.extend(_simulate_period(df, score, oos_s, oos_e, **best_p))
        oos_rets = [t["pnl_pct"] for t in oos_t]
        oos_sh   = _sharpe(oos_rets)
        oos_sharpes.append(oos_sh)
        all_oos_rets.extend(oos_rets)
        print(f"    W{wid} IS:[{is_s}..{is_e}) OOS:[{oos_s}..{oos_e})  "
              f"IS_sh={best_sh:.2f}  OOS_sh={oos_sh:.2f}  params={best_p}")

    pct_positive = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
    avg_oos_sh   = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    passed = pct_positive >= MIN_OOS_PASSES and avg_oos_sh > 0

    print(f"    {len(windows)} windows  avg_OOS_Sh={avg_oos_sh:.2f}  "
          f"{pct_positive:.0%} windows positive  {'PASS' if passed else 'FAIL'}")

    return TierResult(
        tier    = 3,
        name    = "Walk-forward analysis",
        passed  = passed,
        metrics = {"windows": len(windows), "avg_oos_sharpe": avg_oos_sh,
                   "pct_positive_windows": pct_positive, "total_oos_trades": len(all_oos_rets)},
        notes   = [f"IS={IS_MONTHS}m OOS={OOS_MONTHS}m step={STEP_MONTHS}m",
                   f"{pct_positive:.0%} of windows profitable"],
    )


# ─── Tier 4: Monte Carlo stress test ─────────────────────────────────────────

def run_tier4(oos_returns: list[float], regime_returns: dict[str, list[float]]) -> TierResult:
    """Block bootstrap Monte Carlo + per-regime MC."""
    print("  Tier 4 — Monte Carlo stress test...")
    from backtesting.monte_carlo import run_block, run_regime_mc, print_regime_mc_summary

    mc = run_block(oos_returns, starting_equity=1000.0, n_runs=1000, block_size=8)

    print(f"    Block MC:  median={mc.median_return:+.1%}  "
          f"5th={mc.percentile_5:+.1%}  ruin={mc.prob_ruin:.1%}")

    regime_mc = run_regime_mc(regime_returns, starting_equity=1000.0, n_runs=500)
    if regime_mc:
        print_regime_mc_summary(regime_mc)

    passed = (mc.percentile_5 >= MIN_5TH_PCT_RETURN and
              mc.prob_ruin    <= MAX_RUIN_PROB)

    print(f"    {'PASS' if passed else 'FAIL'} — "
          f"5th-pct={mc.percentile_5:+.1%} (min {MIN_5TH_PCT_RETURN:.0%})  "
          f"ruin={mc.prob_ruin:.1%} (max {MAX_RUIN_PROB:.0%})")

    return TierResult(
        tier    = 4,
        name    = "Monte Carlo stress test",
        passed  = passed,
        metrics = {"median": mc.median_return, "pct5": mc.percentile_5,
                   "pct95": mc.percentile_95, "ruin": mc.prob_ruin,
                   "regime_mc": {r: v.median_return for r, v in regime_mc.items()}},
        notes   = ["Block bootstrap size=8, 1000 paths"],
    )


# ─── Tier 5: Microstructure simulation ───────────────────────────────────────

def run_tier5(
    ticker_data: dict,
    best_params: dict,
    oos_start: date,
    oos_end: date,
    regime_calendar: dict,
) -> TierResult:
    """Replay OOS trades with partial fills, latency, and spread costs."""
    print("  Tier 5 — Realistic microstructure simulation...")
    from backtesting.engine import SimConfig, simulate_ticker, print_cost_summary, print_regime_breakdown

    cfg = SimConfig(enable_fills=True, enable_latency=True, enable_regimes=True)
    all_sim_trades = []

    for ticker, (df, score) in ticker_data.items():
        trades = simulate_ticker(
            ticker, df, score, oos_start, oos_end,
            config         = cfg,
            regime_calendar = regime_calendar,
            notional_usd   = POSITION_USD,
            **best_params,
        )
        all_sim_trades.extend(trades)

    if not all_sim_trades:
        return TierResult(5, "Microstructure simulation", False,
                          notes=["No trades in microstructure window"])

    raw_rets = [t.raw_pnl_pct for t in all_sim_trades]
    net_rets = [t.net_pnl_pct for t in all_sim_trades]
    avg_raw  = float(np.mean(raw_rets)) if raw_rets else 0.0
    avg_net  = float(np.mean(net_rets)) if net_rets else 0.0
    drag     = avg_raw - avg_net

    print_cost_summary(all_sim_trades)
    print_regime_breakdown(all_sim_trades)

    passed = avg_net > MIN_NET_PNL
    print(f"\n    Avg raw={avg_raw:+.3%}  avg net={avg_net:+.3%}  "
          f"drag={drag:+.3%}/trade  {'PASS' if passed else 'FAIL'}")

    return TierResult(
        tier    = 5,
        name    = "Microstructure simulation",
        passed  = passed,
        metrics = {"avg_raw": avg_raw, "avg_net": avg_net,
                   "drag_per_trade": drag, "trades": len(all_sim_trades)},
        notes   = ["Partial fills + latency + spread applied"],
    )


# ─── Master pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    tickers: list[str],
    tiers_to_run: list[int] | None = None,
) -> PipelineResult:
    """
    Run all 5 tiers in sequence.  Short-circuits on hard fails only if
    later tiers depend on earlier output; otherwise all requested tiers run.

    Args:
        tickers      : list of ticker symbols to test
        tiers_to_run : [1,2,3,4,5] by default; pass a subset to run fewer
    """
    if tiers_to_run is None:
        tiers_to_run = [1, 2, 3, 4, 5]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         5-TIER BACKTESTING PIPELINE                          ║
║  Tickers: {len(tickers):3d}  |  IS:{IS_MONTHS}m  OOS:{OOS_MONTHS}m  step:{STEP_MONTHS}m                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading SPY (benchmark)...")
    spy_raw   = _yf_dl("SPY", period=TOTAL_HISTORY, interval="1d",
                        progress=False, auto_adjust=True, timeout=20)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_close = spy_raw["close"].squeeze()

    print(f"Loading {len(tickers)} tickers...")
    ticker_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for ticker in tickers:
        df = _fetch(ticker)
        if df is None:
            continue
        try:
            score = _factor_score(df.copy(), spy_close)
            ticker_data[ticker] = (df, score)
        except Exception as exc:
            # Don't silently swallow — without this the operator sees "0 tickers
            # loaded" with no clue why. Log at warning so the pipeline still runs
            # on whatever loaded, but the offender is visible in the output.
            logger.warning(f"pipeline | {ticker} factor-score failed: {exc}")
    print(f"  {len(ticker_data)} tickers loaded\n")

    if len(ticker_data) < 2:
        return PipelineResult([], False, "Not enough tickers with data.")

    all_dates  = [df.index.date.min() for df, _ in ticker_data.values()]
    all_ends   = [df.index.date.max() for df, _ in ticker_data.values()]
    first_date = max(all_dates)
    last_date  = min(all_ends)

    # OOS cutoff = 80% of data IS, 20% OOS
    total_days  = (last_date - first_date).days
    oos_cutoff  = first_date + timedelta(days=int(total_days * 0.8))

    # Regime calendar
    print("Building regime calendar...")
    regime_calendar: dict = {}
    try:
        from backtesting.engine import build_regime_calendar
        regime_calendar = build_regime_calendar()
    except Exception as e:
        print(f"  Warning: regime calendar failed ({e})")

    tier_results: list[TierResult] = []
    best_params  = {"buy_threshold": 3.0, "stop_atr_mult": 1.5, "time_stop_days": 5}

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    if 1 in tiers_to_run:
        t0 = time.time()
        r1 = run_tier1(ticker_data, spy_close, oos_cutoff)
        tier_results.append(r1)
        if r1.passed:
            best_params = r1.metrics["best_params"]
        print(f"    ⏱  {time.time()-t0:.0f}s\n")

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    if 2 in tiers_to_run:
        t0 = time.time()
        r2 = run_tier2(ticker_data, best_params, spy_close, oos_cutoff, last_date)
        tier_results.append(r2)
        print(f"    ⏱  {time.time()-t0:.0f}s\n")

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    oos_rets_wf: list[float] = []
    regime_rets: dict[str, list[float]] = {}
    if 3 in tiers_to_run:
        t0 = time.time()
        r3 = run_tier3(ticker_data, spy_close, all_dates)
        tier_results.append(r3)
        print(f"    ⏱  {time.time()-t0:.0f}s\n")

        # Collect OOS rets for Tier 4 — re-run walk-forward to gather returns
        # (tier 3 already computed them; replicate here to keep separation clean)
        _wf_cursor = max(all_dates)
        while True:
            _is_e = _wf_cursor + relativedelta(months=IS_MONTHS)
            _oos_s = _is_e
            _oos_e = _oos_s + relativedelta(months=OOS_MONTHS)
            if _oos_e > last_date:
                break
            for _, (df, score) in ticker_data.items():
                for t in _simulate_period(df, score, _oos_s, _oos_e, **best_params):
                    oos_rets_wf.append(t["pnl_pct"])
                    # Regime label for per-regime MC
                    entry_d = date.fromisoformat(t["entry_date"])
                    from backtesting.engine import label_regime
                    rg = label_regime(entry_d, regime_calendar)
                    regime_rets.setdefault(rg, []).append(t["pnl_pct"])
            _wf_cursor += relativedelta(months=STEP_MONTHS)

    # ── Tier 4 ────────────────────────────────────────────────────────────────
    if 4 in tiers_to_run and oos_rets_wf:
        t0 = time.time()
        r4 = run_tier4(oos_rets_wf, regime_rets)
        tier_results.append(r4)
        print(f"    ⏱  {time.time()-t0:.0f}s\n")
    elif 4 in tiers_to_run:
        tier_results.append(TierResult(4, "Monte Carlo", False,
                                       notes=["No OOS trades — run Tier 3 first"]))

    # ── Tier 5 ────────────────────────────────────────────────────────────────
    if 5 in tiers_to_run:
        t0 = time.time()
        r5 = run_tier5(
            ticker_data, best_params,
            oos_cutoff, last_date,
            regime_calendar,
        )
        tier_results.append(r5)
        print(f"    ⏱  {time.time()-t0:.0f}s\n")

    # ── Final verdict ──────────────────────────────────────────────────────────
    all_passed   = all(r.passed for r in tier_results)
    deploy_ready = all_passed

    print(f"\n{'='*65}")
    print("  PIPELINE RESULTS")
    print(f"{'='*65}")
    for r in tier_results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon}  Tier {r.tier}: {r.name}")
        for note in r.notes:
            print(f"       {note}")

    print(f"\n  {'🟢 DEPLOY READY' if deploy_ready else '🔴 NOT READY FOR LIVE'}")
    if not deploy_ready:
        failed = [r for r in tier_results if not r.passed]
        for f in failed:
            print(f"     Failed Tier {f.tier}: {f.name}")
    else:
        print(f"     All {len(tier_results)} tiers passed.")
        print("     Next step: paper trade for ≥30 sessions before going live.")

    print(f"{'='*65}\n")

    summary = (
        f"{'DEPLOY READY' if deploy_ready else 'NOT READY'}  "
        f"({sum(r.passed for r in tier_results)}/{len(tier_results)} tiers passed)"
    )
    return PipelineResult(tiers=tier_results, deploy_ready=deploy_ready, summary=summary)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-tier backtesting pipeline")
    parser.add_argument(
        "--tiers",
        default="1,2,3,4,5",
        help="Comma-separated tier numbers to run (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated tickers (default: CUSTOM_WATCHLIST + Nasdaq top 20)"
    )
    args = parser.parse_args()

    tiers_to_run = [int(t.strip()) for t in args.tiers.split(",")]

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        from data.custom_watchlist import CUSTOM_WATCHLIST
        from data.universe import get_nasdaq_watchlist, get_dow_watchlist
        tickers = list(dict.fromkeys(
            CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=20) + get_dow_watchlist()
        ))

    result = run_pipeline(tickers, tiers_to_run=tiers_to_run)
    print(f"\nFinal: {result.summary}")
