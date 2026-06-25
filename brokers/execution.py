"""
brokers/execution.py -- Smart order execution with volatility-adjusted sizing.

Position sizing philosophy
──────────────────────────
Instead of a fixed dollar notional, size is computed from a constant RISK budget
divided by the stock's ATR.  This normalises volatility across all trades so that
a 2%-ATR stock and a 5%-ATR stock carry identical dollar risk at the stop.

    position_notional = risk_per_trade / atr_pct
    e.g. $20 risk / 2% ATR  = $1,000 notional   (calm stock, bigger size)
         $20 risk / 5% ATR  = $400  notional   (volatile stock, smaller size)

Three multipliers then scale the ATR-derived notional:

    regime_mult  – from market_regime.size_mult (vol regime + trend regime)
    quality_mult – from market quality scoring (spread / ADV / slippage)
    breadth_mult – from market breadth filter (SPY/QQQ/RSP/VIX)

    final_notional = atr_notional × regime_mult × quality_mult × breadth_mult

Market Quality    quality_mult
─────────────────────────────
Excellent (5-6)      100%
Moderate  (3-4)       50%
Poor      (1-2)       25%
Reject    (0)          0%  (hard-fail only)

Quality score (0–6) is the sum of three sub-scores:

    Dimension    2 pts (excellent)   1 pt (moderate)    0 pts (poor/hard-fail)
    ─────────────────────────────────────────────────────────────────────────
    Spread       < 0.15%             0.15–0.40%          > 0.40% → hard reject
    Liquidity    ADV > $10M          $1M – $10M          < $1M   → hard reject
    Slippage     < 0.05%             0.05–0.20%          > 0.20% → score 0 (not reject)

Hard rejects (no trade regardless of score):
  - Spread > HARD_MAX_SPREAD_PCT (0.8%)
  - ADV < HARD_MIN_ADV_USD ($5M)   ← raised from $250K after backtest analysis
  - Participation > HARD_MAX_PARTICIPATION (3% of ADV)
  - Price > VWAP + VWAP_MAX_PREMIUM  }  only checked during
  - Price < intraday_high – PULLBACK  }  market hours

Slippage model (Almgren-Chriss square-root form):
    slippage_est = IMPACT_COEFF * atr_pct * sqrt(notional / ADV)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import math
import pandas as pd
from loguru import logger

from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
from utils.metrics import counter as _counter
from utils.retry import with_retry

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hard-reject thresholds  (binary: violating any = no trade)
# ---------------------------------------------------------------------------
HARD_MAX_SPREAD_PCT    = 0.008   # 0.8%  -- market is chaotic / illiquid
HARD_MIN_ADV_USD       = 5_000_000  # $5M -- raised after backtest: micro-caps gap
                                    #        through stops (SLE -28%, CADL -26%, BW -15%)
HARD_MAX_PARTICIPATION = 0.03    # 3% of ADV -- you'd be the market
VWAP_MAX_PREMIUM       = 0.003   # 0.3% above VWAP (intraday only)
PULLBACK_MIN_PCT       = 0.002   # must be ≥0.2% off intraday high

# ---------------------------------------------------------------------------
# Quality scoring thresholds  (soft: scores determine size tier)
# ---------------------------------------------------------------------------
# Spread tiers
SPREAD_EXCELLENT = 0.0015   # < 0.15%  → 2 pts
SPREAD_MODERATE  = 0.004    # < 0.40%  → 1 pt  (above → 0 pts)

# ADV tiers
ADV_EXCELLENT    = 10_000_000  # > $10M  → 2 pts
ADV_MODERATE     = 1_000_000   # > $1M   → 1 pt  (below → 0 pts, hard reject < $250K)

# Slippage tiers
SLIP_EXCELLENT   = 0.0005   # < 0.05%  → 2 pts
SLIP_MODERATE    = 0.002    # < 0.20%  → 1 pt  (above → 0 pts)

# Quality score → size multiplier
QUALITY_SIZE: dict[str, float] = {
    "excellent": 1.00,   # score 5–6
    "moderate":  0.50,   # score 3–4
    "poor":      0.25,   # score 1–2
}

LIMIT_SPREAD_FRAC = 0.35   # limit = bid + LIMIT_SPREAD_FRAC * (ask - bid)

# Square-root market impact calibration constant.
# Formula: slippage = IMPACT_COEFF * atr_pct * sqrt(notional / ADV)
# At 10: a $500 order in a $5M ADV stock with 1.5% ATR → ~0.15% slippage.
IMPACT_COEFF      = 10.0

# ---------------------------------------------------------------------------
# Volatility-adjusted sizing constants
# ---------------------------------------------------------------------------
# Dollar loss tolerance per trade (at 1× ATR stop distance).
# position_notional = RISK_PER_TRADE_USD / atr_pct
# e.g. $20 / 2% ATR = $1,000 notional; $20 / 5% ATR = $400 notional.
# Tune this to your account size and personal risk tolerance.
RISK_PER_TRADE_USD = 20.0

# Hard ceiling per individual position (prevents huge slots on ultra-calm stocks).
MAX_POSITION_USD   = 2_000.0

# Minimum meaningful notional (below this, commissions/spread dominate).
MIN_POSITION_USD   = 25.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

def _score_spread(spread_pct: float) -> int:
    if spread_pct < SPREAD_EXCELLENT: return 2
    if spread_pct < SPREAD_MODERATE:  return 1
    return 0

def _score_adv(adv_usd: float) -> int:
    if adv_usd >= ADV_EXCELLENT: return 2
    if adv_usd >= ADV_MODERATE:  return 1
    return 0

def _score_slippage(slippage_est: float) -> int:
    if slippage_est < SLIP_EXCELLENT: return 2
    if slippage_est < SLIP_MODERATE:  return 1
    return 0

def _quality_tier(score: int) -> str:
    if score >= 5: return "excellent"
    if score >= 3: return "moderate"
    if score >= 1: return "poor"
    return "reject"


@dataclass
class ExecutionContext:
    ticker:           str
    bid:              float
    ask:              float
    mid:              float
    spread_pct:       float
    adv_usd:          float
    atr_pct:          float             # 14-day ATR / close price (volatility proxy)
    participation:    float             # notional / ADV
    vwap:             Optional[float]   # None when market closed / no intraday data
    intraday_high:    Optional[float]
    vwap_diff_pct:    Optional[float]   # (mid - vwap) / vwap
    pullback_pct:     Optional[float]   # (mid - intraday_high) / intraday_high
    slippage_est:     float             # sqrt-model impact estimate
    limit_price:      float
    intraday_ok:      bool              # True when live intraday data was available

    # Computed in __post_init__
    hard_reject:      bool  = field(init=False)
    hard_reject_why:  list  = field(init=False)
    quality_score:    int   = field(init=False)   # 0–6
    quality_tier:     str   = field(init=False)   # excellent/moderate/poor/reject
    size_mult:        float = field(init=False)   # 1.0 / 0.5 / 0.25 / 0.0

    def __post_init__(self) -> None:
        reasons: list[str] = []

        # ── Hard rejects (binary) ───────────────────────────────────────────
        if self.spread_pct >= HARD_MAX_SPREAD_PCT:
            reasons.append(f"spread {self.spread_pct:.2%} ≥ {HARD_MAX_SPREAD_PCT:.1%} hard limit")
        if self.adv_usd < HARD_MIN_ADV_USD:
            reasons.append(f"ADV ${self.adv_usd:,.0f} < ${HARD_MIN_ADV_USD:,.0f} hard limit")
        if self.participation >= HARD_MAX_PARTICIPATION:
            reasons.append(
                f"participation {self.participation:.2%} ≥ {HARD_MAX_PARTICIPATION:.0%} of ADV"
            )
        if self.intraday_ok and self.vwap_diff_pct is not None:
            if self.vwap_diff_pct > VWAP_MAX_PREMIUM:
                reasons.append(f"price {self.vwap_diff_pct:+.2%} above VWAP (max +{VWAP_MAX_PREMIUM:.2%})")
        if self.intraday_ok and self.pullback_pct is not None:
            if self.pullback_pct > -PULLBACK_MIN_PCT:
                reasons.append(
                    f"no pullback: {self.pullback_pct:+.2%} from intraday high (need <{-PULLBACK_MIN_PCT:.2%})"
                )

        self.hard_reject     = bool(reasons)
        self.hard_reject_why = reasons

        # ── Quality scoring (soft — determines size tier) ───────────────────
        if self.hard_reject:
            self.quality_score = 0
        else:
            self.quality_score = (
                _score_spread(self.spread_pct)
                + _score_adv(self.adv_usd)
                + _score_slippage(self.slippage_est)
            )

        self.quality_tier = _quality_tier(self.quality_score)
        self.size_mult    = QUALITY_SIZE.get(self.quality_tier, 0.0)

    @property
    def tradeable(self) -> bool:
        """True if a trade should proceed (at any size tier)."""
        return not self.hard_reject and self.quality_tier != "reject"

    @property
    def rejection_reasons(self) -> list[str]:
        return self.hard_reject_why

    @property
    def quality_summary(self) -> str:
        """One-line quality description for logging."""
        return (
            f"{self.quality_tier.upper()} [{self.quality_score}/6]  "
            f"spread={self.spread_pct:.2%}({_score_spread(self.spread_pct)}pts)  "
            f"ADV=${self.adv_usd/1e6:.1f}M({_score_adv(self.adv_usd)}pts)  "
            f"impact={self.slippage_est:.3%}({_score_slippage(self.slippage_est)}pts)  "
            f"→ size={self.size_mult:.0%}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _alpaca_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _alpaca_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)


@with_retry(max_attempts=2, base_delay=1.0)
def _fetch_quote(ticker: str) -> tuple[float, float]:
    """Return (bid, ask)."""
    from alpaca.data.requests import StockLatestQuoteRequest
    client = _alpaca_data_client()
    q = client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=ticker)
    )[ticker]
    return float(q.bid_price), float(q.ask_price)


@with_retry(max_attempts=2, base_delay=1.0)
def _fetch_intraday(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """Return (vwap, intraday_high) from today's 1-min bars, or (None, None)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    now   = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    client = _alpaca_data_client()
    try:
        bars = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=today,
                end=now,
            )
        )
        df = bars.df
        if df is None or df.empty:
            return None, None
        if hasattr(df.index, "levels"):
            df = df.droplevel(0)
        df.columns = [c.lower() for c in df.columns]
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap    = float((typical * df["volume"]).sum() / df["volume"].sum())
        hi      = float(df["high"].max())
        return vwap, hi
    except Exception:
        return None, None


def _fetch_adv_and_atr(ticker: str) -> tuple[float, float]:
    """Return (20-day avg dollar volume, 14-day ATR as fraction of close).
    Routes through the data layer (Tiingo primary, yfinance fallback).
    Falls back to (0.0, 0.02) on error — 2% ATR is a conservative default.
    """
    try:
        from data.market_data import get_ohlcv as _get_ohlcv
        df = _get_ohlcv(ticker, period="2mo", interval="1d")
        if df is None or df.empty:
            return 0.0, 0.02

        # 20-day ADV
        adv = float((df["close"] * df["volume"]).tail(20).mean())
        adv = adv if adv > 0 else 0.0

        # 14-day ATR (Wilder method)
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14    = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        last_close = float(close.iloc[-1])
        atr_pct  = (atr14 / last_close) if last_close > 0 else 0.02

        return adv, round(atr_pct, 6)
    except Exception:
        return 0.0, 0.02


def _is_market_hours() -> bool:
    """Rough check: US market is open Mon-Fri 09:30-16:00 ET."""
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5:
            return False
        open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_t <= now <= close_t
    except ImportError:
        # pytz not installed -- assume market is open
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_execution_context(ticker: str, notional_usd: float) -> ExecutionContext:
    """Build an ExecutionContext with all pre-flight data for a BUY."""
    bid, ask = _fetch_quote(ticker)
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0

    adv_usd, atr_pct = _fetch_adv_and_atr(ticker)

    # Intraday VWAP + high (only during market hours)
    market_open = _is_market_hours()
    vwap = intraday_high = None
    if market_open:
        vwap, intraday_high = _fetch_intraday(ticker)

    vwap_diff_pct  = (mid - vwap) / vwap if vwap else None
    pullback_pct   = (mid - intraday_high) / intraday_high if intraday_high else None

    # Volume participation rate
    participation = (notional_usd / adv_usd) if adv_usd > 0 else 1.0

    # Nonlinear (square-root) market impact model:
    #   slippage = IMPACT_COEFF * atr_pct * sqrt(participation)
    # - Captures diminishing-returns impact as order size grows
    # - ATR scales impact by realized volatility of the stock
    # - Falls back to worst-case 100% slippage if no ADV data
    if adv_usd > 0:
        slippage_est = IMPACT_COEFF * atr_pct * math.sqrt(participation)
    else:
        slippage_est = 1.0

    # Limit price: bid + LIMIT_SPREAD_FRAC * spread, capped at VWAP when available
    raw_limit = bid + LIMIT_SPREAD_FRAC * (ask - bid)
    if vwap is not None:
        raw_limit = min(raw_limit, vwap)
    # Never above ask, never below bid
    limit_price = round(max(bid, min(raw_limit, ask - 0.01)), 2)

    return ExecutionContext(
        ticker=ticker,
        bid=bid,
        ask=ask,
        mid=round(mid, 4),
        spread_pct=round(spread_pct, 6),
        adv_usd=round(adv_usd, 0),
        atr_pct=round(atr_pct, 6),
        participation=round(participation, 8),
        vwap=round(vwap, 4) if vwap else None,
        intraday_high=round(intraday_high, 4) if intraday_high else None,
        vwap_diff_pct=round(vwap_diff_pct, 6) if vwap_diff_pct is not None else None,
        pullback_pct=round(pullback_pct, 6) if pullback_pct is not None else None,
        slippage_est=round(slippage_est, 6),
        limit_price=limit_price,
        intraday_ok=(market_open and vwap is not None),
    )


def smart_buy(
    ticker: str,
    risk_usd: float = RISK_PER_TRADE_USD,
    *,
    stop_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a smart limit BUY order using volatility-adjusted sizing.

    Sizing pipeline:
        1. atr_notional   = risk_usd / atr_pct         (normalise by volatility)
        2. quality_scaled = atr_notional × quality_mult (penalise poor liquidity)
        3. final_notional = quality_scaled × breadth_mult (penalise weak market)

    Args:
        ticker          : e.g. "AAPL"
        risk_usd        : dollar loss tolerance at 1× ATR stop (default RISK_PER_TRADE_USD).
                          Pass regime.size_mult × RISK_PER_TRADE_USD from the caller to
                          incorporate regime-level sizing on top of ATR normalisation.
        stop_pct        : if set, attaches a stop-loss (e.g. 0.08 = 8% stop)
        take_profit_pct : if set, attaches a take-profit limit

    Returns dict with order id/status, or None if rejected / failed.
    """
    from brokers.alpaca import _already_holding, _trading
    from utils.risk import check_position, daily_limit_hit
    from utils.tradingview import tv as _tv

    if _already_holding(ticker):
        logger.info(f"SKIP {ticker} -- already holding a position.")
        return None

    # Size-independent gate: daily-loss limit / halt check before any market-data calls.
    if daily_limit_hit():
        logger.error(f"SKIP {ticker} -- daily loss limit reached.")
        return None

    # Provisional notional seeds liquidity/participation/slippage pre-flight checks.
    # Uses risk_usd / 2% ATR as a neutral starting estimate; the real ATR-derived
    # size is computed below and re-validated with check_position() before the order.
    seed_notional = min(MAX_POSITION_USD, max(MIN_POSITION_USD, risk_usd / 0.02))
    try:
        ctx = get_execution_context(ticker, seed_notional)
    except Exception as exc:
        logger.warning(f"SKIP {ticker} -- could not build execution context: {exc}")
        return None

    if not ctx.tradeable:
        reasons = "; ".join(ctx.rejection_reasons)
        logger.warning(f"SKIP {ticker} -- hard reject: {reasons}")
        _counter("trades_filtered_total", {"ticker": ticker, "reason": "hard_reject"})
        return None

    # ── Step 1: ATR-based notional ──────────────────────────────────────────
    # Converts a fixed dollar-risk budget into a position size that is
    # proportional to how calm or volatile the stock is.
    if ctx.atr_pct > 0:
        atr_notional = risk_usd / ctx.atr_pct
        atr_notional = max(MIN_POSITION_USD, min(atr_notional, MAX_POSITION_USD))
        sizing_note  = (
            f"ATR-sized: ${risk_usd:.1f}÷{ctx.atr_pct:.2%}"
            f"=${atr_notional:.0f}"
        )
    else:
        # Fallback when ATR unavailable — use a conservative cap
        atr_notional = min(risk_usd * 5, MAX_POSITION_USD)
        sizing_note  = f"ATR fallback: ${atr_notional:.0f}"

    # ── Step 2: breadth multiplier from market regime (cached) ─────────────
    breadth_mult = 1.0
    try:
        from agents.market_regime import compute_regime
        regime = compute_regime()
        breadth_mult = regime.breadth_mult
    except Exception as _be:
        logger.debug(f"Breadth filter unavailable ({_be}) -- using 1.0x")

    # ── Step 3: apply quality tier + breadth on top of ATR notional ────────
    sized_notional = atr_notional * ctx.size_mult * breadth_mult
    sized_notional = max(MIN_POSITION_USD, sized_notional)

    logger.info(
        f"{ticker} sizing | {sizing_note}  "
        f"quality={ctx.size_mult:.0%}({ctx.quality_tier})  "
        f"breadth={breadth_mult:.2f}x  "
        f"=> final ${sized_notional:.0f}"
    )
    logger.info(f"{ticker} quality {ctx.quality_summary}")

    # Final risk gate on the ACTUAL sized notional (position cap + daily limit).
    if not check_position(sized_notional):
        return None

    # Fractional qty from sized notional / limit_price
    qty = round(sized_notional / ctx.limit_price, 4)
    if qty <= 0:
        logger.warning(f"SKIP {ticker} -- computed qty {qty} <= 0.")
        return None

    try:
        if stop_pct is not None:
            # Bracket/stop limit orders require whole shares
            qty = max(1, math.floor(qty))
            stop_price   = round(ctx.limit_price * (1 - stop_pct), 2)
            from alpaca.trading.requests import (
                TakeProfitRequest, StopLossRequest, LimitOrderRequest
            )
            from alpaca.trading.enums import OrderClass
            kwargs: dict = dict(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                type="limit",
                time_in_force=TimeInForce.DAY,
                limit_price=ctx.limit_price,
                order_class=OrderClass.SIMPLE,
                stop_loss=StopLossRequest(stop_price=stop_price),
            )
            if take_profit_pct is not None:
                tp_price = round(ctx.limit_price * (1 + take_profit_pct), 2)
                kwargs["take_profit"] = TakeProfitRequest(limit_price=tp_price)
                kwargs["order_class"] = OrderClass.BRACKET
            req = LimitOrderRequest(**kwargs)
        else:
            req = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=ctx.limit_price,
            )

        order = _trading.submit_order(req)

        vwap_str = f"  VWAP_diff={ctx.vwap_diff_pct:+.2%}" if ctx.vwap_diff_pct is not None else ""
        pb_str   = f"  pullback={ctx.pullback_pct:+.2%}"    if ctx.pullback_pct is not None else ""
        logger.info(
            f"LIMIT BUY {ticker} x{qty} @ ${ctx.limit_price:.2f}  "
            f"[{ctx.quality_tier.upper()} {ctx.size_mult:.0%}]  "
            f"spread={ctx.spread_pct:.2%}  ADV=${ctx.adv_usd/1e6:.1f}M  "
            f"ATR={ctx.atr_pct:.2%}  impact={ctx.slippage_est:.3%}"
            f"{vwap_str}{pb_str}  order_id={order.id}"
        )
        _counter("trades_placed_total", {"side": "BUY", "type": "limit", "market": "stock"})
        _tv.after_trade(ticker)
        return {"id": str(order.id), "status": str(order.status),
                "limit_price": ctx.limit_price, "qty": qty}

    except Exception as exc:
        if "not fractionable" in str(exc) or "40310000" in str(exc):
            logger.warning(f"SKIP {ticker} -- not fractionable on Alpaca.")
        else:
            logger.error(f"LIMIT BUY {ticker} failed: {exc}")
        return None
