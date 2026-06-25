"""
data/finnhub_data.py — FinnHub REST API wrapper.

Provides three capabilities that complement the existing data layer:

  1. News sentiment  — get_news_sentiment(ticker)
     Fetches the last 7 days of company news and returns a composite
     sentiment score (-1.0 bearish … +1.0 bullish).  Used as a quality
     gate before Claude AI Stage-2 scoring and in the IB screener.

  2. Social sentiment — get_social_sentiment(ticker)
     FinnHub aggregates Reddit/StockTwits buzz and bull/bear scores.

  3. Earnings surprise — get_earnings_surprise(ticker)
     Last 4 quarters of EPS actual vs estimate.  Positive streak → green
     flag.  Negative miss on most recent quarter → yellow/red flag.

  4. Basic financials — get_basic_financials(ticker)
     PE ratio, revenue growth, gross margin, etc. for fundamental gates.

All functions:
  - Are fail-soft (return None / empty on any error)
  - Cache results in memory with appropriate TTLs
  - Require FINNHUB_API_KEY in .env

Free tier limits: 60 calls/min, no WebSocket.
"""
from __future__ import annotations

import os
import time
import threading
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from loguru import logger


# ---------------------------------------------------------------------------
# Session + auth
# ---------------------------------------------------------------------------

_BASE = "https://finnhub.io/api/v1"
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


class FinnHubDisabledError(RuntimeError):
    """Raised when FINNHUB_API_KEY is not configured."""


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        key = os.environ.get("FINNHUB_API_KEY", "")
        if not key:
            try:
                from config import FINNHUB_API_KEY
                key = FINNHUB_API_KEY
            except ImportError:
                pass
        if not key:
            raise FinnHubDisabledError(
                "FINNHUB_API_KEY is not set.  Add it to .env to enable FinnHub data."
            )
        s = requests.Session()
        s.headers.update({"X-Finnhub-Token": key})
        _SESSION = s
        logger.info("FinnHub session initialised (key: ...{})", key[-4:])
        return _SESSION


def _get(path: str, params: dict | None = None, timeout: int = 10) -> dict | list:
    url = f"{_BASE}{path}"
    resp = _get_session().get(url, params=params or {}, timeout=timeout)
    if resp.status_code == 401:
        raise PermissionError(f"FinnHub 401: invalid API key")
    if resp.status_code == 429:
        raise ConnectionError(f"FinnHub rate-limit (429)")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str, ttl_sec: float) -> object | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < ttl_sec:
        return entry[1]
    return None


def _cache_set(key: str, value: object) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# 1. Company news + sentiment composite
# ---------------------------------------------------------------------------

_NEWS_TTL_SEC = 30 * 60   # refresh every 30 minutes


def get_company_news(ticker: str, days: int = 3) -> list[dict]:
    """
    Return recent company news articles from FinnHub.

    Each item: {datetime, headline, summary, source, url, sentiment}
    where sentiment is 'positive' | 'negative' | 'neutral' (FinnHub field).

    Args:
        ticker : stock symbol
        days   : how many calendar days back to fetch (default 3)
    """
    key = f"news:{ticker.upper()}:{days}"
    cached = _cache_get(key, _NEWS_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore

    try:
        end   = date.today()
        start = end - timedelta(days=days)
        raw = _get("/company-news", {
            "symbol": ticker.upper(),
            "from":   start.isoformat(),
            "to":     end.isoformat(),
        })
        if not isinstance(raw, list):
            return []
        articles = [
            {
                "datetime": a.get("datetime", 0),
                "headline": a.get("headline", ""),
                "summary":  a.get("summary", ""),
                "source":   a.get("source", ""),
                "url":      a.get("url", ""),
                "sentiment": a.get("sentiment", "neutral"),
            }
            for a in raw[:20]   # cap at 20 articles
        ]
        _cache_set(key, articles)
        logger.debug("FinnHub | {} news: {} articles ({}d)", ticker, len(articles), days)
        return articles
    except FinnHubDisabledError:
        raise
    except Exception as exc:
        logger.warning("FinnHub | get_company_news({}) failed: {}", ticker, exc)
        return []


def get_news_sentiment_score(ticker: str, days: int = 3) -> float:
    """
    Composite news sentiment score in [-1.0, +1.0].

    Positive headlines outweigh → score > 0
    Negative headlines outweigh → score < 0
    No news or neutral mix      → 0.0

    Used as a quality gate: score < -0.3 suppresses auto-trade.
    """
    key = f"news_score:{ticker.upper()}:{days}"
    cached = _cache_get(key, _NEWS_TTL_SEC)
    if cached is not None:
        return float(cached)

    articles = get_company_news(ticker, days=days)
    if not articles:
        return 0.0

    _sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
    scores = [_sentiment_map.get(a.get("sentiment", "neutral"), 0) for a in articles]
    composite = sum(scores) / len(scores) if scores else 0.0
    composite = round(max(-1.0, min(1.0, composite)), 3)
    _cache_set(key, composite)
    logger.debug("FinnHub | {} news sentiment: {:.3f} ({} articles)", ticker, composite, len(articles))
    return composite


# ---------------------------------------------------------------------------
# 2. Social sentiment (Reddit / StockTwits)
# ---------------------------------------------------------------------------

_SOCIAL_TTL_SEC = 60 * 60   # 1 hour


def get_social_sentiment(ticker: str) -> dict:
    """
    FinnHub social sentiment aggregate.

    Returns:
        {
            "buzz":       float,  # relative mention volume (1.0 = average)
            "bull_score": float,  # 0-1 fraction bullish
            "bear_score": float,  # 0-1 fraction bearish
            "score":      float,  # bull_score - bear_score  (-1 to +1)
        }
    Returns {} on failure.
    """
    key = f"social:{ticker.upper()}"
    cached = _cache_get(key, _SOCIAL_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore

    try:
        raw = _get("/stock/social-sentiment", {"symbol": ticker.upper(), "from": (date.today() - timedelta(days=7)).isoformat()})
        if not isinstance(raw, dict):
            return {}

        # FinnHub returns reddit + twitter sections
        reddit  = raw.get("reddit", [{}])[0]  if raw.get("reddit")  else {}
        twitter = raw.get("twitter", [{}])[0] if raw.get("twitter") else {}

        bull = (reddit.get("positiveMention", 0) + twitter.get("positiveMention", 0))
        bear = (reddit.get("negativeMention", 0) + twitter.get("negativeMention", 0))
        total = bull + bear

        result = {
            "buzz":       raw.get("buzz", {}).get("weeklyAverage", 1.0),
            "bull_score": bull / total if total > 0 else 0.5,
            "bear_score": bear / total if total > 0 else 0.5,
            "score":      round((bull - bear) / total, 3) if total > 0 else 0.0,
        }
        _cache_set(key, result)
        logger.debug("FinnHub | {} social: score={:.3f} buzz={:.2f}", ticker, result["score"], result["buzz"])
        return result
    except FinnHubDisabledError:
        raise
    except Exception as exc:
        logger.warning("FinnHub | get_social_sentiment({}) failed: {}", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# 3. Earnings surprise history
# ---------------------------------------------------------------------------

_EARNINGS_TTL_SEC = 24 * 3600


def get_earnings_surprise(ticker: str) -> list[dict]:
    """
    Return last 4 quarters of EPS actual vs estimate.

    Each item: {period, actual, estimate, surprise_pct}
    Sorted most-recent first.

    Positive surprise_pct = beat estimates.
    """
    key = f"eps:{ticker.upper()}"
    cached = _cache_get(key, _EARNINGS_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore

    try:
        raw = _get("/stock/earnings", {"symbol": ticker.upper(), "limit": 4})
        if not isinstance(raw, list):
            return []
        result = []
        for item in raw:
            actual   = item.get("actual")
            estimate = item.get("estimate")
            if actual is None or estimate is None or estimate == 0:
                surprise = None
            else:
                surprise = round((actual - estimate) / abs(estimate) * 100, 1)
            result.append({
                "period":       item.get("period", ""),
                "actual":       actual,
                "estimate":     estimate,
                "surprise_pct": surprise,
            })
        _cache_set(key, result)
        logger.debug("FinnHub | {} earnings: {} quarters", ticker, len(result))
        return result
    except FinnHubDisabledError:
        raise
    except Exception as exc:
        logger.warning("FinnHub | get_earnings_surprise({}) failed: {}", ticker, exc)
        return []


def earnings_quality_flag(ticker: str) -> str:
    """
    Summarise earnings surprise history as a flag string.

    Returns:
        "GREEN"  — beat estimates in 3+ of last 4 quarters
        "YELLOW" — mixed (2 beats)
        "RED"    — missed most-recent quarter OR beat < 2 of last 4
        "NONE"   — no data available
    """
    surprises = get_earnings_surprise(ticker)
    if not surprises:
        return "NONE"

    # Most recent miss → RED immediately
    if surprises[0].get("surprise_pct") is not None and surprises[0]["surprise_pct"] < 0:
        return "RED"

    beats = sum(1 for s in surprises if s.get("surprise_pct") is not None and s["surprise_pct"] >= 0)
    if beats >= 3:
        return "GREEN"
    if beats == 2:
        return "YELLOW"
    return "RED"


# ---------------------------------------------------------------------------
# 4. Basic financials
# ---------------------------------------------------------------------------

_FINANCIALS_TTL_SEC = 6 * 3600   # 6 hours


def get_basic_financials(ticker: str) -> dict:
    """
    Return key fundamental metrics from FinnHub.

    Useful fields: peNormalizedAnnual, revenueGrowthTTMYoy, grossMarginTTM,
                   52WeekHigh, 52WeekLow, beta, marketCapitalization.
    Returns {} on failure.
    """
    key = f"fin:{ticker.upper()}"
    cached = _cache_get(key, _FINANCIALS_TTL_SEC)
    if cached is not None:
        return cached  # type: ignore

    try:
        raw = _get("/stock/metric", {"symbol": ticker.upper(), "metric": "all"})
        metric = raw.get("metric", {}) if isinstance(raw, dict) else {}
        result = {k: v for k, v in metric.items() if v is not None}
        _cache_set(key, result)
        logger.debug("FinnHub | {} financials: {} fields", ticker, len(result))
        return result
    except FinnHubDisabledError:
        raise
    except Exception as exc:
        logger.warning("FinnHub | get_basic_financials({}) failed: {}", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# 5. Composite signal quality score
# ---------------------------------------------------------------------------

def get_signal_quality(ticker: str) -> dict:
    """
    Combine news sentiment, social sentiment, and earnings quality into a
    single signal-quality dict.  Used in Stage 2 scoring to boost or suppress
    a technical signal.

    Returns:
        {
            "news_score":     float,   # -1 to +1
            "social_score":  float,   # -1 to +1
            "earnings_flag": str,     # GREEN / YELLOW / RED / NONE
            "composite":     float,   # weighted average, -1 to +1
            "gate":          str,     # PASS / WATCH / BLOCK
        }

    Gate logic:
        PASS  — composite >= -0.1   (neutral or better)
        WATCH — composite < -0.1    (mild negative, reduce position size)
        BLOCK — composite < -0.4 OR earnings_flag == RED
    """
    news    = get_news_sentiment_score(ticker)
    social  = get_social_sentiment(ticker).get("score", 0.0)
    earn    = earnings_quality_flag(ticker)

    earn_num = {"GREEN": 0.3, "YELLOW": 0.0, "RED": -0.4, "NONE": 0.0}[earn]
    composite = round(0.4 * news + 0.3 * social + 0.3 * earn_num, 3)

    if earn == "RED" or composite < -0.4:
        gate = "BLOCK"
    elif composite < -0.1:
        gate = "WATCH"
    else:
        gate = "PASS"

    return {
        "news_score":    news,
        "social_score":  social,
        "earnings_flag": earn,
        "composite":     composite,
        "gate":          gate,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG")

    for ticker in ["NVDA", "PLTR", "AAPL"]:
        print(f"\n=== {ticker} ===")
        q = get_signal_quality(ticker)
        print(f"  News:     {q['news_score']:+.3f}")
        print(f"  Social:   {q['social_score']:+.3f}")
        print(f"  Earnings: {q['earnings_flag']}")
        print(f"  Composite:{q['composite']:+.3f}  Gate: {q['gate']}")
