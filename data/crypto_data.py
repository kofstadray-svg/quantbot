"""
data/crypto_data.py — Fetch crypto OHLCV data.

Primary source: Tiingo crypto endpoint (when TIINGO_API_KEY is set).
Fallback:       yfinance (BTC-USD format, free, no auth required).

Crypto symbols in yfinance/Tiingo use BTC-USD format.
Alpaca uses BTC/USD format — conversion handled in brokers/alpaca.py.
"""
from __future__ import annotations
import pandas as pd
import ta
import yfinance as yf
from loguru import logger
from config import TIINGO_ENABLED

try:
    from data.tiingo_data import get_crypto_ohlcv as _tiingo_crypto, TiingoDisabledError
    _TIINGO_AVAILABLE = True
except ImportError:
    _TIINGO_AVAILABLE = False
    TiingoDisabledError = Exception  # type: ignore[assignment,misc]

# Default crypto watchlist
# ── Only tickers confirmed active on Alpaca paper trading ──────────────────
# Removed: TON-USD, DOT-USD (crash: "asset not found" on Alpaca → kills scheduler)
# Removed: BNB-USD, TRX-USD, NEAR-USD, ICP-USD, ETC-USD, XLM-USD, ATOM-USD
#          (not in Alpaca's active crypto catalog — would crash if screener fires)
CRYPTO_WATCHLIST = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "SOL-USD",   # Solana
    "XRP-USD",   # XRP
    "DOGE-USD",  # Dogecoin
    "AVAX-USD",  # Avalanche
    "LINK-USD",  # Chainlink
    "ADA-USD",   # Cardano
    "LTC-USD",   # Litecoin
    "BCH-USD",   # Bitcoin Cash
    "UNI-USD",   # Uniswap
    "AAVE-USD",  # Aave
    "ARB-USD",   # Arbitrum
]


def get_crypto_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV for a crypto symbol (e.g. 'BTC-USD').
    Uses Tiingo when TIINGO_API_KEY is set; falls back to yfinance.
    """
    # --- Tiingo (primary) ---
    if _TIINGO_AVAILABLE and TIINGO_ENABLED:
        try:
            df = _tiingo_crypto(symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                return df
            logger.warning("Tiingo crypto | {} empty, falling back to yfinance", symbol)
        except TiingoDisabledError:
            pass
        except Exception as exc:
            logger.warning("Tiingo crypto | {} error ({}), falling back to yfinance", symbol, exc)

    # --- yfinance (fallback) ---
    logger.info("yfinance crypto fallback | {} period={}", symbol, period)
    df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=20)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def compute_crypto_indicators(symbol: str) -> dict:
    """Compute all screener indicators for a crypto asset."""
    df = get_crypto_ohlcv(symbol, period="1y")
    if df.empty or len(df) < 50:
        return {}

    close  = df["close"].squeeze()
    volume = df["volume"].squeeze()

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    # MACD histogram
    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]

    # Moving averages
    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    golden_cross = bool(ma200 and ma50 > ma200)

    # Relative volume
    avg_vol = volume.rolling(20).mean().iloc[-1]
    rel_volume = round(float(volume.iloc[-1] / avg_vol), 2) if avg_vol else 1.0

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_range = bb_upper - bb_lower
    bb_position = round(float((close.iloc[-1] - bb_lower) / bb_range), 3) if bb_range else 0.5

    # 7-day and 30-day price change (important for crypto momentum)
    price_change_7d  = round(float((close.iloc[-1] - close.iloc[-7])  / close.iloc[-7]),  4)
    price_change_30d = round(float((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]), 4)

    return {
        "symbol":          symbol.replace("-USD", ""),
        "price":           round(float(close.iloc[-1]), 4),
        "rsi_14":          round(float(rsi), 2),
        "macd_signal":     round(float(macd_hist), 6),
        "golden_cross":    golden_cross,
        "rel_volume":      rel_volume,
        "bb_position":     bb_position,
        "price_change_7d": price_change_7d,
        "price_change_30d": price_change_30d,
    }
