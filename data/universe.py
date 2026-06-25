"""
data/universe.py — Dynamically fetch stock universes from major indexes.

Indexes supported:
  - S&P 600  (small cap)
  - S&P 400  (mid cap)
  - Nasdaq 100
  - Dow Jones Industrial Average (30 stocks)

Pulls components from Wikipedia, then pre-filters by volume and price
so only the most active names reach Claude.
"""
from __future__ import annotations
import io
import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from utils import cache as _cache

# Index component lists change very rarely, so a cached list stays valid for
# weeks. 30 days keeps the full ~600-name universe available through any
# Wikipedia outage, instead of collapsing to the tiny hardcoded stub.
_UNIVERSE_CACHE_MAX_AGE_SEC = 30 * 24 * 3600


def _resolve_index(cache_key: str, fetch_fn, fallback: list[str], label: str) -> list[str]:
    """
    Cache-aware index resolver:
      1. try the live fetch; on success cache the full list and return it
      2. on failure, return the last cached full list if it's not too old
      3. only if there's no usable cache, fall back to the hardcoded stub
    This means a Wikipedia hiccup keeps the FULL universe (from cache) rather
    than dropping to the ~100-name stub.
    """
    try:
        tickers = fetch_fn()
        if tickers:
            _cache.save(cache_key, {"tickers": tickers}, label=label)
            return tickers
    except Exception as e:
        logger.warning(f"{label}: live fetch failed ({type(e).__name__}: {e})")
    entry = _cache.load(cache_key, max_age_sec=_UNIVERSE_CACHE_MAX_AGE_SEC)
    if entry is not None:
        tk = entry["payload"].get("tickers", [])
        if tk:
            logger.warning(
                f"{label}: serving CACHED universe ({len(tk)} names, "
                f"{entry['age_sec']/3600:.0f}h old) — Wikipedia unavailable."
            )
            return tk
    logger.warning(f"{label}: no live data and no cache — using hardcoded fallback ({len(fallback)}).")
    return fallback[:]


# Wikipedia blocks the default urllib user-agent that pandas.read_html uses
# (HTTP 403). Fetch the page ourselves with a browser UA, then parse the HTML
# string. Requires lxml to be installed.
_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _read_wiki_tables(url: str, header: int = 0) -> list[pd.DataFrame]:
    """Fetch a Wikipedia page with a browser UA and return its HTML tables."""
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text), header=header)


# ─── Index sources (Wikipedia) ────────────────────────────────────────────────

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","PEP","INTC","INTU","CSCO","CMCSA","TMUS",
    "TXN","AMGN","HON","SBUX","AMAT","LRCX","MDLZ","GILD","REGN","VRTX",
    "PYPL","MU","KLAC","SNPS","CDNS","MRVL","PANW","CRWD","WDAY","ORLY",
    "ASML","ADP","ROST","KDP","CHTR","ABNB","MNST","DXCM","CTAS","ODFL",
    "MCHP","FTNT","PAYX","CPRT","PCAR","MRNA","KHC","AEP","BIIB","IDXX",
    "FAST","EA","BKR","XEL","GEHC","ON","GFS","DDOG","TEAM","ZS",
    "TTD","ILMN","VRSK","DLTR","FANG","ENPH","ALGN","JD","PDD",
    "BIDU","NTES","LCID","RIVN","CEG","EXC","CTSH","MELI","ISRG",
    "LULU","ZM","OKTA","NXPI","SWKS","QRVO","CDW","NTAP","SIRI","TROW",
]


def _get_nasdaq100() -> list[str]:
    """Nasdaq 100 components: live Wikipedia → cached → hardcoded fallback."""
    def _fetch():
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100", header=0)
        for df in tables:
            cols = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()]
            if cols:
                tickers = df[cols[0]].dropna().str.replace(".", "-", regex=False).tolist()
                logger.info(f"Nasdaq 100 loaded: {len(tickers)} tickers from Wikipedia")
                return tickers
        return []
    return _resolve_index("universe_nasdaq100", _fetch, _NASDAQ100_FALLBACK, "Nasdaq 100")


def _get_dow30() -> list[str]:
    """Dow Jones 30 components — hardcoded since it rarely changes."""
    tickers = [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
    ]
    logger.info(f"Dow 30 loaded: {len(tickers)} tickers")
    return tickers


# Representative S&P 600 small-caps (top names by market cap / liquidity)
_SP600_FALLBACK = [
    "ACLS","AEIS","AGIO","ALKS","AMBC","AMED","AMKR","AMSF","APOG","AROC",
    "ASTE","ATRC","AVAV","BANF","BCPC","BDC","BHE","BKE","BLKB","BRC",
    "BRKR","CABO","CADE","CALM","CATO","CBRL","CBU","CCRN","CEVA","CHCO",
    "CHEF","CHRS","CLFD","CLW","CMPR","CNO","COHU","COLB","CPK","CRGY",
    "CRVL","CSR","CTBI","CTRE","CTS","CULP","CXM","DCOM","DECK","DIOD",
    "DJCO","DLX","DOCN","DRH","EFC","EFSC","ELSE","ENS","EPAC","ESE",
    "ESNT","ETD","EVTC","EXP","EXPO","FCNCA","FFIN","FISI","FLIC","FLXS",
    "FORM","FOXF","FRME","FRPH","FULT","GKOS","GOLF","GOOG","HAIN","HARL",
    "HCI","HCSG","HIBB","HLF","HMST","HNRG","HONE","HTH","HUBG","HWKN",
    "ICFI","ICHR","IIIN","IOSP","IPAR","IRWD","JACK","JBSS","JELD","JJSF",
]

# Representative S&P 400 mid-caps
_SP400_FALLBACK = [
    "AAN","ACC","ACHC","ADNT","AFG","AGCO","AHH","AIT","AL","ALSN",
    "AMG","AMKR","AN","ANF","AOS","APA","APG","APPF","ARW","ASGN",
    "ATI","AVNT","AWI","AYI","BCC","BCPE","BDN","BFH","BHF","BJ",
    "BKH","BMI","BOX","BRC","BXMT","CABO","CACI","CADE","CALM","CATY",
    "CBT","CBTS","CC","CCCS","CCS","CDK","CIEN","CNA","CNX","COG",
    "COLM","CPT","CR","CRC","CROX","CRUS","CSL","CTRA","DAN","DAVA",
    "DCI","DCOMP","DCP","DDS","DKS","DLB","DLX","DNOW","DOCS","DORM",
    "DPZ","DRQ","DT","DTM","DXPE","EAT","EHC","EIG","EME","ENOV",
    "EPR","EQT","ESI","ESRT","EXR","FAF","FBP","FCFS","FHN","FICO",
    "FLS","FMTB","FNF","FORM","FR","FRPT","GATX","GEF","GFF","GHC",
]


def _get_sp600() -> list[str]:
    """S&P 600 Small Cap: live Wikipedia → cached → hardcoded fallback."""
    def _fetch():
        tables = _read_wiki_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", header=0)
        df = tables[0]
        col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        tickers = df[col].dropna().str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 600 loaded: {len(tickers)} tickers from Wikipedia")
        return tickers
    return _resolve_index("universe_sp600", _fetch, _SP600_FALLBACK, "S&P 600")


def _get_sp400() -> list[str]:
    """S&P 400 Mid Cap: live Wikipedia → cached → hardcoded fallback."""
    def _fetch():
        tables = _read_wiki_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", header=0)
        df = tables[0]
        col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        tickers = df[col].dropna().str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 400 loaded: {len(tickers)} tickers from Wikipedia")
        return tickers
    return _resolve_index("universe_sp400", _fetch, _SP400_FALLBACK, "S&P 400")


# ─── Pre-filter: keep only the most active names ─────────────────────────────

def _filter_by_activity(
    tickers: list[str],
    min_price: float = 5.0,
    min_avg_volume: int = 200_000,
    top_n: int = 50,
) -> list[str]:
    """
    Download 5-day summary for the full list (one fast batch call),
    then keep only liquid stocks above a minimum price.
    Returns the top_n by relative volume — the ones moving today.
    """
    if not tickers:
        return []

    logger.info(f"Pre-filtering {len(tickers)} tickers (batch download)…")
    try:
        raw = yf.download(
            tickers,
            period="5d",
            interval="1d",
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=20,
        )
    except Exception as e:
        logger.error(f"Batch download failed: {e}")
        return tickers[:top_n]

    rows = []
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][ticker].dropna()
                volume = raw["Volume"][ticker].dropna()
            else:
                close = raw["Close"].dropna()
                volume = raw["Volume"].dropna()

            if close.empty or volume.empty:
                continue

            price = float(close.iloc[-1])
            avg_vol = float(volume.mean())
            today_vol = float(volume.iloc[-1])
            rel_vol = today_vol / avg_vol if avg_vol > 0 else 0

            if price < min_price or avg_vol < min_avg_volume:
                continue

            rows.append({"ticker": ticker, "price": price, "rel_volume": rel_vol})
        except Exception:
            continue

    if not rows:
        return tickers[:top_n]

    df = pd.DataFrame(rows).sort_values("rel_volume", ascending=False)
    result = df["ticker"].head(top_n).tolist()
    logger.info(f"Pre-filter kept {len(result)} tickers (top by relative volume)")
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def get_nasdaq_watchlist(top_n: int = 30) -> list[str]:
    """Top top_n most active Nasdaq 100 stocks today."""
    return _filter_by_activity(_get_nasdaq100(), top_n=top_n)


def get_dow_watchlist() -> list[str]:
    """All 30 Dow Jones stocks (small enough to screen every one)."""
    return _get_dow30()


def get_small_cap_watchlist(top_n: int = 30) -> list[str]:
    """Top top_n most active S&P 600 small cap stocks today."""
    return _filter_by_activity(_get_sp600(), top_n=top_n)


def get_mid_cap_watchlist(top_n: int = 30) -> list[str]:
    """Top top_n most active S&P 400 mid cap stocks today."""
    return _filter_by_activity(_get_sp400(), top_n=top_n)


def get_small_and_mid_cap_watchlist(top_n: int = 40) -> list[str]:
    """Combined small + mid cap universe, top_n most active names."""
    return _filter_by_activity(_get_sp600() + _get_sp400(), top_n=top_n)


def get_full_watchlist(top_n_each: int = 20) -> list[str]:
    """
    All four indexes combined — Dow 30 (all) + top movers from
    Nasdaq 100, S&P 400, and S&P 600.
    Deduplicates so overlapping tickers aren't screened twice.
    """
    dow = _get_dow30()
    nasdaq = _filter_by_activity(_get_nasdaq100(), top_n=top_n_each)
    mid = _filter_by_activity(_get_sp400(), top_n=top_n_each)
    small = _filter_by_activity(_get_sp600(), top_n=top_n_each)
    combined = list(dict.fromkeys(dow + nasdaq + mid + small))  # deduplicate, keep order
    logger.info(f"Full watchlist: {len(combined)} unique tickers across all indexes")
    return combined


if __name__ == "__main__":
    print("\n── Dow 30 ──────────────────────────────")
    print(", ".join(get_dow_watchlist()))

    print("\n── Top 10 Nasdaq 100 (by activity) ────")
    print(", ".join(get_nasdaq_watchlist(top_n=10)))

    print("\n── Top 10 Small Cap (by activity) ─────")
    print(", ".join(get_small_cap_watchlist(top_n=10)))

    print("\n── Top 10 Mid Cap (by activity) ────────")
    print(", ".join(get_mid_cap_watchlist(top_n=10)))
