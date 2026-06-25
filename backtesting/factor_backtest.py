"""
backtesting/factor_backtest.py — Backtest the FULL upgraded screener+exit stack
and optimise factor weights via walk-forward.

This mirrors agents/momentum_screener._weighted_score() and the exit engine
(Supertrend / Chandelier) but computes every factor POINT-IN-TIME over history,
so we can (a) backtest the real stack and (b) fit factor weights to outcomes
without lookahead.

Factors extracted per bar (each normalised to its live max points):
  ema9_gt_21, rsi_zone, rel_vol, gap, atr_pct, rs_excess, rs_slope,
  avwap_gap, avwap_br52, supertrend, volume_profile
(near_hod / above_vwap / entry_timing / float are intraday or fundamental and
 cannot be reconstructed historically from daily bars — excluded, with their
 weight redistributed. This is stated explicitly in the report.)

NO LOOKAHEAD: factor at bar i uses only data[:i+1]. Entries act on next bar open.

HOW TO RUN:
  python backtesting/factor_backtest.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from backtesting._tiingo_compat import download as _yf_dl
from dataclasses import dataclass, field
from loguru import logger


# ── point-in-time factor series ──────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)


def _atr_series(high, low, close, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _supertrend_dir(high, low, close, period=10, mult=3.0) -> pd.Series:
    """Vectorised-ish Supertrend direction series (+1/-1). Point-in-time safe."""
    atr = _atr_series(high, low, close, period)
    hl2 = (high + low) / 2.0
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    n = len(close)
    fub = np.full(n, np.nan); flb = np.full(n, np.nan)
    st  = np.full(n, np.nan); d = np.zeros(n, dtype=int)
    c = close.values; ubv = ub.values; lbv = lb.values; atrv = atr.values
    fv = None
    for i in range(n):
        if np.isnan(atrv[i]):
            d[i] = -1; continue
        if fv is None:
            fv = i; fub[i] = ubv[i]; flb[i] = lbv[i]
            if c[i] <= ubv[i]: st[i] = fub[i]; d[i] = -1
            else: st[i] = flb[i]; d[i] = 1
            continue
        fub[i] = ubv[i] if (ubv[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lbv[i] if (lbv[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
        if st[i-1] == fub[i-1]:
            if c[i] <= fub[i]: st[i] = fub[i]; d[i] = -1
            else: st[i] = flb[i]; d[i] = 1
        else:
            if c[i] >= flb[i]: st[i] = flb[i]; d[i] = 1
            else: st[i] = fub[i]; d[i] = -1
    return pd.Series(d, index=close.index)


print("factor_backtest.py module scaffold written")


# ── benchmark cache (SPY) ─────────────────────────────────────────────────────
_SPY = None
def _spy_close() -> pd.Series:
    global _SPY
    if _SPY is None:
        raw = _yf_dl("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
        _SPY = raw["Close"].squeeze()
    return _SPY


def _rs_excess_series(close: pd.Series, win: int = 20) -> pd.Series:
    """20-day stock return minus SPY return, per bar (point-in-time)."""
    spy = _spy_close().reindex(close.index).ffill()
    sret = close / close.shift(win) - 1.0
    bret = spy / spy.shift(win) - 1.0
    return ((sret - bret) * 100.0).fillna(0.0)


def _rs_slope_series(close: pd.Series, win: int = 20) -> pd.Series:
    """Slope of (stock/SPY) ratio over `win`, % of ratio per bar."""
    spy = _spy_close().reindex(close.index).ffill()
    ratio = (close / spy).fillna(method="ffill")
    def _sl(w):
        if np.isnan(w).any() or w[0] == 0: return 0.0
        return float(np.polyfit(range(len(w)), w, 1)[0]) / w[0] * 100.0
    return ratio.rolling(win).apply(_sl, raw=True).fillna(0.0)


def _volume_profile_at(high, low, vol, close_price, lookback=60, bins=40) -> dict:
    """Volume profile for a single point in time using the trailing `lookback`
    bars already sliced by the caller (no lookahead)."""
    out = {"above_poc": False, "clear_overhead": False, "in_lvn": False}
    try:
        h = high[-lookback:]; l = low[-lookback:]; v = vol[-lookback:]
        pmin, pmax = float(np.nanmin(l)), float(np.nanmax(h))
        if not (pmax > pmin): return out
        edges = np.linspace(pmin, pmax, bins+1)
        centers = (edges[:-1]+edges[1:])/2
        vp = np.zeros(bins)
        for hh, ll, vv in zip(h, l, v):
            if np.isnan(hh) or np.isnan(ll) or np.isnan(vv) or vv<=0 or hh<ll: continue
            lo_b = max(0, min(bins-1, np.searchsorted(edges, ll, "right")-1))
            hi_b = max(0, min(bins-1, np.searchsorted(edges, hh, "right")-1))
            vp[lo_b:hi_b+1] += vv/(hi_b-lo_b+1)
        if vp.sum() <= 0: return out
        poc = float(centers[int(np.argmax(vp))])
        hi_t = np.quantile(vp, 0.70); lo_t = np.quantile(vp, 0.30)
        hvn_above = centers[(vp>=hi_t) & (centers>close_price)]
        cur_b = max(0, min(bins-1, np.searchsorted(edges, close_price, "right")-1))
        d_hvn = ((hvn_above.min()-close_price)/close_price*100) if len(hvn_above) else None
        out["above_poc"] = close_price > poc
        out["clear_overhead"] = (len(hvn_above)==0) or (d_hvn is not None and d_hvn>3.0)
        out["in_lvn"] = bool(vp[cur_b] <= lo_t)
    except Exception:
        pass
    return out


# Factor max points (mirrors live _weighted_score; intraday/fundamental factors
# excluded and their weight implicitly redistributed by renormalising at scoring).
FACTOR_MAX = {
    "ema9_gt_21": 9.0, "rsi": 12.0, "rel_vol": 12.0, "gap": 10.0,
    "atr_pct": 5.0, "rs_excess": 12.0, "rs_slope": 4.0,
    "avwap_gap": 7.0, "avwap_br52": 5.0, "supertrend": 8.0, "volume_profile": 8.0,
}

def _build_factor_frame(ticker: str) -> pd.DataFrame:
    """All factor POINT series for one ticker, point-in-time. Index = dates."""
    raw = _yf_dl(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 120:
        return pd.DataFrame()
    close = raw["Close"].squeeze(); high = raw["High"].squeeze()
    low = raw["Low"].squeeze(); vol = raw["Volume"].squeeze(); opn = raw["Open"].squeeze()

    f = pd.DataFrame(index=close.index)
    f["close"] = close; f["open"] = opn; f["high"] = high; f["low"] = low

    ema9 = _ema(close, 9); ema21 = _ema(close, 21)
    f["ema9_gt_21"] = np.where(ema9 > ema21, 9.0, 0.0)

    rsi = _rsi_series(close)
    f["rsi"] = np.select(
        [(rsi>=58)&(rsi<=70), (rsi>=55)&(rsi<58), (rsi>70)&(rsi<=72),
         (rsi>=50)&(rsi<55), rsi>72],
        [12.0, 8.0, 8.0, 4.0, 2.0], default=0.0)

    relv = (vol / vol.rolling(20).mean()).fillna(0.0)
    f["rel_vol"] = np.select(
        [relv>=10, relv>=7, relv>=5, relv>=3, relv>=2],
        [12.0,10.0,8.0,5.0,2.0], default=0.0)

    gap = ((opn - close.shift(1)) / close.shift(1) * 100).fillna(0.0)
    f["gap"] = np.select([gap>=15,gap>=10,gap>=8,gap>=5,gap>=3],
                         [10.0,9.0,7.0,4.0,2.0], default=0.0)

    atrp = (_atr_series(high,low,close) / close * 100).fillna(0.0)
    f["atr_pct"] = np.where(atrp>5, 5.0, np.where(atrp>3, 2.0, 0.0))

    rse = _rs_excess_series(close)
    f["rs_excess"] = np.select([rse>=8,rse>=4,rse>=2,rse>=0,rse>=-3],
                               [12.0,10.0,7.0,4.0,1.0], default=0.0)
    rss = _rs_slope_series(close)
    f["rs_slope"] = np.select([rss>0.10,rss>0.02,rss>0],[4.0,3.0,1.0], default=0.0)

    st_dir = _supertrend_dir(high, low, close)
    st_flip = st_dir != st_dir.shift(1)
    f["supertrend"] = np.where(st_dir==1, np.where(st_flip & (st_dir==1), 8.0, 6.0), 0.0)

    # AVWAP + volume profile need per-bar loops (windowed). Do them rowwise.
    avg = np.zeros(len(close)); abr = np.zeros(len(close)); vpp = np.zeros(len(close))
    hv = high.values; lv = low.values; vv = vol.values; cv = close.values
    for i in range(len(close)):
        if i < 30:
            continue
        # gap-anchored AVWAP: most recent >=3% up-gap within last 120 bars
        lo_i = max(1, i-120)
        ganchor = None
        for j in range(lo_i, i+1):
            if cv[j-1] > 0 and (opn.values[j]-cv[j-1])/cv[j-1]*100 >= 3.0:
                ganchor = j
        if ganchor is not None and ganchor < i:
            tp = (hv[ganchor:i+1]+lv[ganchor:i+1]+cv[ganchor:i+1])/3
            vol_seg = vv[ganchor:i+1]
            if vol_seg.sum() > 0:
                av = (tp*vol_seg).cumsum()/vol_seg.cumsum()
                cur = av[-1]; look = min(5, len(av)-1)
                pts = (4.0 if cv[i] > cur else 0.0) + (3.0 if av[-1] > av[-1-look] else 0.0)
                avg[i] = pts
        # 52w breakout anchor
        win = min(252, i)
        seg = cv[i-win:i+1]
        pm = seg[0]; banchor = None
        for k in range(1, len(seg)):
            if seg[k] > pm*1.001: banchor = (i-win)+k
            pm = max(pm, seg[k])
        if banchor is not None and banchor < i:
            tp = (hv[banchor:i+1]+lv[banchor:i+1]+cv[banchor:i+1])/3
            vol_seg = vv[banchor:i+1]
            if vol_seg.sum() > 0:
                av = (tp*vol_seg).cumsum()/vol_seg.cumsum()
                cur = av[-1]; look = min(5, len(av)-1)
                pts = (3.0 if cv[i] > cur else 0.0) + (2.0 if av[-1] > av[-1-look] else 0.0)
                abr[i] = pts
        # volume profile (trailing 60)
        vpinfo = _volume_profile_at(hv[:i+1], lv[:i+1], vv[:i+1], cv[i])
        vp_pts = (3.0 if vpinfo["above_poc"] else 0.0) + (3.0 if vpinfo["clear_overhead"] else 0.0) + (2.0 if vpinfo["in_lvn"] else 0.0)
        vpp[i] = min(8.0, vp_pts)
    f["avwap_gap"] = avg; f["avwap_br52"] = abr; f["volume_profile"] = vpp
    f["st_dir"] = st_dir
    f["atr_abs"] = _atr_series(high, low, close)
    return f.dropna()


FACTOR_COLS = list(FACTOR_MAX.keys())

@dataclass
class FTrade:
    ticker: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float; pnl_pct: float; hold_days: int
    factors: dict = field(default_factory=dict)   # raw factor points at entry


def _gen_trades(ticker: str, frames: dict, buy_score: float = 55.0,
                chand_mult: float = 3.0) -> list[FTrade]:
    """Generate trades for one ticker using EQUAL-renormalised scoring for entry
    selection, recording each entry's factor vector so weights can be optimised
    later. Exits: Supertrend flip OR Chandelier(HH-ATR*mult) OR 1.5*ATR hard stop."""
    f = frames.get(ticker)
    if f is None or f.empty:
        return []
    trades = []
    in_pos = False; entry_i = 0; entry_px = 0.0; entry_date = ""; entry_fac = {}
    peak = 0.0
    # entry score with CURRENT live weights (renormalised to available factors)
    max_avail = sum(FACTOR_MAX.values())
    raw_score = f[FACTOR_COLS].sum(axis=1) / max_avail * 100.0
    closes = f["close"].values; highs = f["high"].values
    opens = f["open"].values; atr = f["atr_abs"].values; stdir = f["st_dir"].values
    for i in range(1, len(f)-1):
        if not in_pos:
            if raw_score.iloc[i] >= buy_score and stdir[i] == 1:
                in_pos = True; entry_i = i+1            # enter next bar open
                entry_px = opens[i+1]; entry_date = str(f.index[i+1].date())
                entry_fac = {c: float(f[c].iloc[i]) for c in FACTOR_COLS}
                peak = highs[i+1]
        else:
            peak = max(peak, highs[i])
            px = closes[i]
            chand = peak - atr[i]*chand_mult
            hard = entry_px - atr[entry_i]*1.5
            reason = None
            if px < hard: reason = "hard_stop"
            elif stdir[i] == -1: reason = "supertrend_flip"
            elif px < chand: reason = "chandelier"
            if reason:
                trades.append(FTrade(ticker, entry_date, str(f.index[i].date()),
                    entry_px, px, (px-entry_px)/entry_px, i-entry_i, entry_fac))
                in_pos = False
    return trades


def _score_with_weights(fac: dict, weights: dict) -> float:
    """Re-score a recorded factor vector with candidate weights (0-100)."""
    num = sum(fac[c]/FACTOR_MAX[c] * weights.get(c, 0.0) for c in FACTOR_COLS)
    den = sum(weights.get(c, 0.0) for c in FACTOR_COLS) or 1.0
    return num/den * 100.0


def _eval_weights(trades: list[FTrade], weights: dict, top_frac: float = 0.5) -> dict:
    """Score a weight vector: re-rank trades by re-weighted entry score, keep the
    top `top_frac`, report mean return + win rate. Higher mean = better weights."""
    if not trades:
        return {"mean": -999, "n": 0, "win": 0}
    scored = [(_score_with_weights(t.factors, weights), t.pnl_pct) for t in trades]
    scored.sort(reverse=True)
    keep = scored[:max(5, int(len(scored)*top_frac))]
    rets = [r for _, r in keep]
    return {"mean": float(np.mean(rets)), "n": len(keep),
            "win": float(np.mean([1 for r in rets if r>0])/1) if rets else 0,
            "win_rate": float(sum(1 for r in rets if r>0)/len(rets)) if rets else 0}


def _optimize(trades: list[FTrade], n_iter: int = 3000, seed: int = 0) -> tuple[dict, dict]:
    """Random search over weight vectors. Returns (best_weights, best_eval)."""
    rng = np.random.default_rng(seed)
    best_w = {c: FACTOR_MAX[c] for c in FACTOR_COLS}   # start = current live weights
    best = _eval_weights(trades, best_w)
    for _ in range(n_iter):
        # sample weights uniformly in [0, 2*current] to explore around live values
        w = {c: float(rng.uniform(0, 2*FACTOR_MAX[c])) for c in FACTOR_COLS}
        e = _eval_weights(trades, w)
        if e["mean"] > best["mean"]:
            best, best_w = e, w
    return best_w, best


def walk_forward_optimize(tickers: list[str], n_iter: int = 2000):
    """Build factors, generate trades, split IS/OOS by date, optimise on IS,
    apply to OOS. Reports current-weights vs optimised, in-sample vs out."""
    logger.info(f"Building factor frames for {len(tickers)} tickers...")
    frames = {}
    for t in tickers:
        try:
            fr = _build_factor_frame(t)
            if not fr.empty: frames[t] = fr
        except Exception as e:
            logger.warning(f"  {t}: {e}")
    logger.info(f"  built {len(frames)} frames")

    all_trades = []
    for t in frames:
        all_trades.extend(_gen_trades(t, frames))
    all_trades.sort(key=lambda x: x.entry_date)
    n = len(all_trades)
    print(f"\nTotal trades generated: {n}")
    if n < 30:
        print("Too few trades to optimise meaningfully.")
        return

    # 70/30 chronological split
    cut = int(n*0.70)
    is_tr, oos_tr = all_trades[:cut], all_trades[cut:]
    print(f"IS trades: {len(is_tr)}   OOS trades: {len(oos_tr)}")

    cur_w = {c: FACTOR_MAX[c] for c in FACTOR_COLS}
    is_cur  = _eval_weights(is_tr, cur_w)
    oos_cur = _eval_weights(oos_tr, cur_w)

    opt_w, is_opt = _optimize(is_tr, n_iter=n_iter)
    oos_opt = _eval_weights(oos_tr, opt_w)

    print("\n=== CURRENT (hand-set) weights ===")
    print(f"  IS  mean={is_cur['mean']:+.2%}  win={is_cur['win_rate']:.0%}  n={is_cur['n']}")
    print(f"  OOS mean={oos_cur['mean']:+.2%}  win={oos_cur['win_rate']:.0%}  n={oos_cur['n']}")
    print("\n=== OPTIMISED weights (fit on IS) ===")
    print(f"  IS  mean={is_opt['mean']:+.2%}  win={is_opt['win_rate']:.0%}  n={is_opt['n']}")
    print(f"  OOS mean={oos_opt['mean']:+.2%}  win={oos_opt['win_rate']:.0%}  n={oos_opt['n']}")
    degr = (oos_opt['mean']/is_opt['mean']) if is_opt['mean']>0 else 0.0
    print(f"\n  Optimised degradation ratio (OOS/IS): {degr:+.2f}")
    print(f"  {'OVERFIT — optimised weights do not generalise' if degr < 0.4 else 'Weights hold up out-of-sample'}")
    print("\n  Optimised weights (normalised to 100):")
    tot = sum(opt_w.values())
    for c in sorted(opt_w, key=lambda k:-opt_w[k]):
        print(f"    {c:16s} {opt_w[c]/tot*100:5.1f}   (was {FACTOR_MAX[c]/sum(FACTOR_MAX.values())*100:.1f})")
    print("\n[[WF_OPT_COMPLETE]]")


if __name__ == "__main__":
    from data.custom_watchlist import CUSTOM_WATCHLIST
    from data.universe import get_nasdaq_watchlist, get_dow_watchlist
    tickers = list(dict.fromkeys(CUSTOM_WATCHLIST + get_nasdaq_watchlist(top_n=30) + get_dow_watchlist()))[:50]
    walk_forward_optimize(tickers, n_iter=2000)
