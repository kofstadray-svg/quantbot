"""
backtesting/engine.py -- Realistic simulation engine with microstructure.

Three layers sit on top of the basic signal→trade loop:

1. PARTIAL FILLS
   ─────────────
   • Volume participation cap: orders > PART_CAP_PCT of daily volume are
     pro-rated. e.g. $1,000 order in a $50,000 ADV stock = 2% participation;
     with a 1% cap the fill is only 50% of shares wanted.
   • Spread cost: estimated from ATR (illiquid stocks have wider spreads).
     Entry at ask  = mid + half_spread
     Exit  at bid  = mid - half_spread
   • Round-trip spread drag reduces every trade's P&L.

2. LATENCY SIMULATION
   ───────────────────
   • Scanner fires at bar close, but the order reaches the market 1–3 bars
     later (10–30 min round-trip including decision + routing latency).
   • Price movement during latency: Gaussian per-bar drift with a slight
     adverse-selection bias (market tends to move away from you once you
     decide to buy — other fast participants already acted on the signal).
   • Entry price = close + latency_drift + market_impact(atr, participation).

3. REGIME SEGMENTATION
   ────────────────────
   • Each trade is tagged with the market regime at its entry date:
       bull        SPY > 50 EMA  AND  VIX < 20
       bear        SPY < 50 EMA
       high_vol    VIX 20–30
       extreme_vol VIX > 30
       neutral     everything else
   • regime_breakdown() slices win rate, Sharpe, and expectancy by regime.

Import pattern:
    from backtesting.engine import SimConfig, simulate_ticker, regime_breakdown
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import ta
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

# ─── Regime calendar cache ────────────────────────────────────────────────────
_REGIME_CACHE: dict[date, str] = {}
_REGIME_LOADED = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All knobs for the realistic simulation engine.

    Toggle individual layers to compare idealized vs realistic backtest:
        idealized  : enable_fills=False, enable_latency=False, enable_regimes=False
        fills_only : enable_fills=True,  enable_latency=False
        full       : enable_fills=True,  enable_latency=True,  enable_regimes=True
    """
    # Partial fill parameters
    enable_fills:      bool  = True
    part_cap_pct:      float = 0.01     # max 1% of daily volume per order
    spread_atr_coeff:  float = 0.15     # spread ≈ 15% of ATR  (empirical)

    # Latency parameters
    enable_latency:    bool  = True
    latency_bars_min:  int   = 1        # minimum bars before fill (next bar)
    latency_bars_max:  int   = 3        # maximum bars (30-min scanner cycle)
    adverse_sel_coeff: float = 0.10     # mean adverse drift per bar as frac of bar_vol
    impact_coeff:      float = 10.0     # Almgren-Chriss calibration constant

    # Regime labeling (tags each trade — no execution impact)
    enable_regimes:    bool  = True

    # Random seed (set to None for stochastic runs)
    seed:              Optional[int] = 42


# ---------------------------------------------------------------------------
# Regime calendar
# ---------------------------------------------------------------------------

def build_regime_calendar(force: bool = False) -> dict[date, str]:
    """
    Fetch SPY and VIX once and build a {date → regime_label} dict.

    Regimes:
        bull        SPY > 50 EMA  and  VIX < 20
        bear        SPY < 50 EMA
        high_vol    VIX 20–30  (overrides bull/neutral)
        extreme_vol VIX > 30   (overrides everything)
        neutral     otherwise
    """
    global _REGIME_CACHE, _REGIME_LOADED
    if _REGIME_LOADED and not force:
        return _REGIME_CACHE

    try:
        spy_raw = _yf_dl("SPY", period="5y", interval="1d",
                          progress=False, auto_adjust=True, timeout=20)
        vix_raw = _yf_dl("^VIX", period="5y", interval="1d",
                          progress=False, auto_adjust=True, timeout=20)

        for df in [spy_raw, vix_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]

        spy_c = spy_raw["close"].squeeze()
        vix_c = vix_raw["close"].squeeze()

        ema50 = spy_c.ewm(span=50, adjust=False).mean()
        above_50 = spy_c > ema50

        cal: dict[date, str] = {}
        for ts in spy_c.index:
            d   = ts.date()
            ab  = bool(above_50.loc[ts])
            vix = float(vix_c.reindex([ts]).ffill().iloc[0]) if ts in vix_c.index else 20.0

            if vix > 30:
                label = "extreme_vol"
            elif vix > 20:
                label = "high_vol"
            elif ab:
                label = "bull"
            else:
                label = "bear"
            cal[d] = label

        _REGIME_CACHE  = cal
        _REGIME_LOADED = True
        return cal
    except Exception:
        # Fallback: empty calendar — trades will be labeled "unknown"
        _REGIME_CACHE  = {}
        _REGIME_LOADED = True
        return {}


def label_regime(entry_date: date, calendar: dict[date, str]) -> str:
    """Return regime label for a given trade entry date."""
    if not calendar:
        return "unknown"
    # Look backwards up to 5 trading days to handle gaps (weekends, holidays)
    for delta in range(6):
        d = entry_date - timedelta(days=delta)
        if d in calendar:
            return calendar[d]
    return "unknown"


# ---------------------------------------------------------------------------
# Microstructure helpers
# ---------------------------------------------------------------------------

def _estimate_adv(df: pd.DataFrame, row_idx: int, window: int = 20) -> float:
    """20-day average dollar volume ending at row_idx."""
    start = max(0, row_idx - window)
    sub   = df.iloc[start:row_idx]
    if sub.empty:
        return 1_000_000.0  # default $1M ADV
    return float((sub["close"] * sub["volume"]).mean())


def _estimate_spread(atr_pct: float, coeff: float) -> float:
    """Bid-ask spread as fraction: spread ≈ coeff × ATR%."""
    return max(0.0005, min(0.008, coeff * atr_pct))


def simulate_fill(
    price: float,
    notional_usd: float,
    adv_usd: float,
    atr_pct: float,
    config: SimConfig,
    rng: np.random.Generator,
    *,
    is_entry: bool = True,
) -> tuple[float, float]:
    """
    Simulate realistic order execution.

    Returns:
        (executed_price, fill_fraction)
        executed_price  : price actually paid (includes spread, no latency)
        fill_fraction   : fraction of desired notional actually filled (0–1)
    """
    if not config.enable_fills:
        return price, 1.0

    # Volume participation
    if adv_usd > 0 and price > 0:
        order_shares  = notional_usd / price
        daily_shares  = adv_usd / price
        participation = order_shares / max(daily_shares, 1.0)
    else:
        participation = 0.0

    # Partial fill if over the cap
    if participation > config.part_cap_pct and participation > 0:
        fill_fraction = config.part_cap_pct / participation
    else:
        fill_fraction = 1.0

    # Spread cost
    spread_pct     = _estimate_spread(atr_pct, config.spread_atr_coeff)
    half_spread    = spread_pct / 2.0
    executed_price = price * (1 + half_spread) if is_entry else price * (1 - half_spread)

    return executed_price, fill_fraction


def simulate_latency(
    entry_price: float,
    atr_pct: float,
    adv_usd: float,
    notional_usd: float,
    config: SimConfig,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """
    Model scanner-to-fill latency and return (adjusted_entry_price, n_bars_delayed).

    Price movement per latency bar:
        bar_vol = daily_vol / sqrt(bars_per_day)
        drift   = N(adverse_sel_coeff × bar_vol,  bar_vol)
    Market impact on execution:
        impact  = impact_coeff × atr_pct × sqrt(participation)
    """
    if not config.enable_latency:
        return entry_price, 0

    BARS_PER_DAY = 39   # 10-min bars in a 6.5-hr session

    n_bars = int(rng.integers(config.latency_bars_min, config.latency_bars_max + 1))

    daily_vol = atr_pct / math.sqrt(252)
    bar_vol   = daily_vol / math.sqrt(BARS_PER_DAY)

    # Adverse drift: slight positive mean (price moves up as we try to buy)
    total_drift = float(rng.normal(
        loc   = config.adverse_sel_coeff * bar_vol * n_bars,
        scale = bar_vol * math.sqrt(n_bars),
    ))

    # Market impact (Almgren-Chriss square-root model)
    if adv_usd > 0:
        participation = notional_usd / adv_usd
        impact = config.impact_coeff * atr_pct * math.sqrt(participation)
    else:
        impact = 0.0

    total_slippage  = total_drift + impact
    adjusted_price  = entry_price * (1 + total_slippage)

    return adjusted_price, n_bars


# ---------------------------------------------------------------------------
# Trade data class
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    ticker:         str
    entry_date:     str
    exit_date:      str
    entry_price:    float     # idealized entry (bar close)
    exit_price:     float     # idealized exit  (bar close)
    actual_entry:   float     # after fills + latency
    actual_exit:    float     # after fills
    raw_pnl_pct:    float     # idealized: (exit - entry) / entry
    net_pnl_pct:    float     # realistic: (actual_exit - actual_entry) / actual_entry
    fill_fraction:  float     # fraction of position actually filled
    latency_bars:   int       # bars of delay on entry
    latency_cost:   float     # % drag from latency
    spread_cost:    float     # % drag from bid-ask (round trip)
    hold_days:      int
    exit_reason:    str
    regime:         str = "unknown"   # market regime at entry
    window_id:      int = 0
    period:         str = "IS"        # "IS" or "OOS"


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate_ticker(
    ticker: str,
    df: pd.DataFrame,           # must have columns: open high low close volume _atr
    score: pd.Series,           # factor score aligned with df.index
    start: date,
    end: date,
    buy_threshold: float,
    stop_atr_mult: float,
    time_stop_days: int,
    config: SimConfig,
    regime_calendar: dict[date, str],
    *,
    window_id: int = 0,
    period: str = "IS",
    notional_usd: float = 100.0,
) -> list[SimTrade]:
    """
    Simulate trades on df[start:end] with microstructure layers applied.

    Args:
        ticker          : ticker symbol
        df              : OHLCV + _atr column
        score           : factor score series
        start, end      : date range to simulate
        buy_threshold   : factor score trigger
        stop_atr_mult   : hard stop = entry_price - mult × ATR
        time_stop_days  : max holding days
        config          : SimConfig with all knobs
        regime_calendar : output of build_regime_calendar()
        notional_usd    : position size for ADV participation calculation
    """
    rng = np.random.default_rng(config.seed)

    mask  = (df.index.date >= start) & (df.index.date < end)
    sub   = df.loc[mask].copy()
    sub_s = score.loc[mask]

    if len(sub) < 5:
        return []

    trades: list[SimTrade] = []
    in_pos = False
    entry_price = actual_entry = entry_atr = 0.0
    fill_frac = latency_bars_used = 0
    latency_cost = spread_cost = 0.0
    entry_idx  = 0
    entry_date_str = ""

    for i in range(1, len(sub)):
        idx     = sub.index[i]
        dt      = idx.date()
        s       = float(sub_s.iloc[i])
        close   = float(sub["close"].iloc[i])
        atr_val = float(sub["_atr"].iloc[i]) if not np.isnan(sub.get("_atr", pd.Series()).iloc[i] if "_atr" in sub.columns else float("nan")) else close * 0.02
        # Safer ATR extraction
        if "_atr" in sub.columns:
            raw_atr = sub["_atr"].iloc[i]
            atr_val = float(raw_atr) if not (isinstance(raw_atr, float) and np.isnan(raw_atr)) else close * 0.02
        else:
            atr_val = close * 0.02

        atr_pct = atr_val / close if close > 0 else 0.02
        adv_usd = _estimate_adv(sub, i)

        if not in_pos:
            if s >= buy_threshold:
                # ── Simulate entry ──────────────────────────────────────────
                exec_price, frac = simulate_fill(
                    close, notional_usd, adv_usd, atr_pct, config, rng, is_entry=True
                )
                lat_price, n_lat = simulate_latency(
                    exec_price, atr_pct, adv_usd, notional_usd, config, rng
                )

                in_pos        = True
                entry_price   = close        # idealized
                actual_entry  = lat_price    # realistic
                entry_atr     = atr_val if atr_val > 0 else close * 0.02
                entry_idx     = i
                fill_frac     = frac
                latency_bars_used = n_lat
                latency_cost  = (lat_price - exec_price) / exec_price if exec_price > 0 else 0.0
                spread_cost   = (exec_price - close) / close if close > 0 else 0.0
                entry_date_str = str(dt)

        else:
            hold   = i - entry_idx
            reason = None

            # Hard ATR stop
            stop_level = actual_entry - stop_atr_mult * entry_atr
            if close < stop_level:
                reason = f"hard_stop({stop_atr_mult}ATR)"

            # Time stop
            elif hold >= time_stop_days:
                rsi_val = float(
                    ta.momentum.RSIIndicator(sub["close"].iloc[:i+1], window=14)
                    .rsi().iloc[-1]
                )
                ema5 = float(sub["close"].iloc[max(0, i-4):i+1].mean())
                if rsi_val < 50 and close < ema5:
                    reason = f"time_stop({time_stop_days}d)"

            # Score reversal
            if reason is None and s < -1.0:
                reason = "score_exit"

            if reason:
                # ── Simulate exit ───────────────────────────────────────────
                exec_exit, _ = simulate_fill(
                    close, notional_usd * fill_frac, adv_usd, atr_pct,
                    config, rng, is_entry=False
                )
                # Exit spread cost (sell at bid) already baked into exec_exit

                raw_pnl  = (close - entry_price) / entry_price
                net_pnl  = (exec_exit - actual_entry) / actual_entry
                exit_spread = (close - exec_exit) / close if close > 0 else 0.0
                rt_spread   = spread_cost + exit_spread

                regime = label_regime(
                    date.fromisoformat(entry_date_str), regime_calendar
                ) if config.enable_regimes else "disabled"

                trades.append(SimTrade(
                    ticker        = ticker,
                    entry_date    = entry_date_str,
                    exit_date     = str(dt),
                    entry_price   = entry_price,
                    exit_price    = close,
                    actual_entry  = actual_entry,
                    actual_exit   = exec_exit,
                    raw_pnl_pct   = raw_pnl,
                    net_pnl_pct   = net_pnl * fill_frac,  # scale by fill
                    fill_fraction = fill_frac,
                    latency_bars  = latency_bars_used,
                    latency_cost  = latency_cost,
                    spread_cost   = rt_spread,
                    hold_days     = hold,
                    exit_reason   = reason,
                    regime        = regime,
                    window_id     = window_id,
                    period        = period,
                ))
                in_pos = False

    return trades


# ---------------------------------------------------------------------------
# Regime-segmented reporting
# ---------------------------------------------------------------------------

def regime_breakdown(trades: list[SimTrade]) -> dict:
    """
    Slice trade metrics by market regime.

    Returns dict: regime → {count, win_rate, avg_raw, avg_net, drag, sharpe}
    """
    from collections import defaultdict
    buckets: dict[str, list[SimTrade]] = defaultdict(list)
    for t in trades:
        buckets[t.regime].append(t)

    results = {}
    order   = ["bull", "neutral", "high_vol", "extreme_vol", "bear", "unknown"]
    for regime in order + [r for r in buckets if r not in order]:
        bucket = buckets.get(regime, [])
        if not bucket:
            continue
        raw    = [t.raw_pnl_pct for t in bucket]
        net    = [t.net_pnl_pct for t in bucket]
        wins   = [r for r in net if r > 0]
        drag   = [t.latency_cost + t.spread_cost for t in bucket]
        sh     = float(np.mean(net) / np.std(net) * np.sqrt(252)) if np.std(net) > 0 else 0.0

        results[regime] = {
            "count":    len(bucket),
            "win_rate": len(wins) / len(net) if net else 0.0,
            "avg_raw":  float(np.mean(raw)),
            "avg_net":  float(np.mean(net)),
            "avg_drag": float(np.mean(drag)),
            "sharpe":   sh,
        }
    return results


def print_regime_breakdown(trades: list[SimTrade]) -> None:
    bk = regime_breakdown(trades)
    if not bk:
        return
    print(f"\n  REGIME SEGMENTATION  ({len(trades)} trades)")
    print(f"  {'Regime':<14}  {'Trades':>6}  {'WinRate':>7}  {'AvgRaw':>7}  {'AvgNet':>7}  {'Drag':>6}  {'Sharpe':>6}")
    print(f"  {'─'*66}")
    for regime, m in bk.items():
        print(
            f"  {regime:<14}  {m['count']:>6}  {m['win_rate']:>6.0%}  "
            f"  {m['avg_raw']:>+6.1%}  {m['avg_net']:>+6.1%}  "
            f"{m['avg_drag']:>+5.2%}  {m['sharpe']:>6.2f}"
        )


# ---------------------------------------------------------------------------
# Microstructure cost summary
# ---------------------------------------------------------------------------

def print_cost_summary(trades: list[SimTrade]) -> None:
    """Compare raw vs net P&L to quantify the total cost of realistic execution."""
    if not trades:
        return
    raw_ret  = [t.raw_pnl_pct for t in trades]
    net_ret  = [t.net_pnl_pct for t in trades]
    lat_cost = [t.latency_cost for t in trades]
    spr_cost = [t.spread_cost  for t in trades]
    fills    = [t.fill_fraction for t in trades]
    partial  = [t for t in trades if t.fill_fraction < 0.999]

    print(f"""
  MICROSTRUCTURE COST ANALYSIS  ({len(trades)} trades)
  {'─'*55}
  Avg raw P&L (idealized)   : {np.mean(raw_ret):>+.3%}
  Avg net P&L (realistic)   : {np.mean(net_ret):>+.3%}
  Total execution drag      : {np.mean(raw_ret)-np.mean(net_ret):>+.3%} per trade

  Avg latency cost          : {np.mean(lat_cost):>+.3%}  ({np.mean([t.latency_bars for t in trades]):.1f} bars avg)
  Avg spread cost (r/t)     : {np.mean(spr_cost):>+.3%}
  Partial fills             : {len(partial)} trades ({len(partial)/len(trades):.0%})  avg fill={np.mean(fills):.0%}
  {'─'*55}
  Edge survival             : {"YES — strategy profitable after costs" if np.mean(net_ret) > 0 else "NO — costs kill the edge"}""")
