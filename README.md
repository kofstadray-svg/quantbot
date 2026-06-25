# QuantBot — Autonomous Multi-Strategy Python Trading Bot

> **Paper-trading first. Production-ready architecture.**
> Three parallel alpha engines, event-based entries, ATR-scaled exits, Claude AI signal scoring, Telegram control, and a full Monte Carlo backtesting suite.

---

## Overview

QuantBot is a fully autonomous algorithmic trading system built in Python. It runs continuously, scans the market multiple times per day across three independent strategy engines, routes signals through a conflict-resolution layer, manages positions with strategy-specific exit profiles, and reports everything through Telegram.

The system is designed around one core principle: **every design decision must be attributable**. Each trade is tagged to its strategy, each exit to its reason, and each signal to the indicator that fired it — so you always know exactly what is generating alpha and what isn't.

---

## Key Features

- **Three parallel alpha engines** running simultaneously with independent signal logic
- **Event-based entries** — all conditions must align; no score-stacking or correlated indicators
- **ATR-scaled exit profiles** per strategy — 2 ATR partial, 4 ATR partial, Chandelier trail
- **Claude AI Stage-2 scoring** — shortlisted tickers get a narrative quality gate before execution
- **Signal router** — deduplicates signals across engines with explicit conflict-resolution rules
- **Market regime model** — SPY slope, ADX, QQQ leadership, VIX breadth; gates all entries
- **Telegram command interface** — on-demand scans, position P&L, strategy attribution
- **TradingView webhook bridge** — receive TradingView alerts and auto-trade through the bot
- **FinnHub webhook receiver** — real-time news events and earnings alerts
- **Flask performance dashboard** — live metrics, open positions, trade journal
- **Full backtesting suite** — per-strategy Monte Carlo, walk-forward OOS validation
- **Preflight diagnostics** — bot refuses to start unless all data providers pass health checks
- **Watchdog process** — auto-restarts crashed services; global mutex prevents double-launches
- **Strategy attribution** — SQLite trade journal with `performance_by_strategy()` query
- **Data layer resilience** — Alpaca primary → Tiingo secondary → yfinance tertiary, 30-min cache

---

## Strategy Engines

### Setup A — Small-Cap Expansion *(in development)*
Low-float, high-relative-volume momentum explosions. The highest reward-to-risk setup.
- Float < 50M shares
- Relative volume > 5× average
- Price above 20-day anchored VWAP
- New 20-day high
- ADX > 20
- RS vs SPY > 10%

Expected profile: low win rate, large winners, 5R–10R+ potential.
Capital allocation: 20–30%.

---

### Setup B — Institutional Breakout *(live)*
Large-cap and mid-cap breakouts with institutional accumulation signatures. The core strategy.
- RS Rank ≥ 70 (top 30% of S&P 500 + Nasdaq 100 universe)
- ADX > 20 (trend established)
- Price above earnings-anchored AVWAP (63-day proxy fallback)
- 20-day avg volume > 120% of 60-day avg volume
- Close above 30-day high

**Exit profile (ATR-based):**
| Level | Trigger | Action |
|-------|---------|--------|
| Initial stop | Entry − 2 ATR | Full exit if breached |
| TP1 | Entry + 2 ATR | Sell 50% |
| TP2 | Entry + 4 ATR | Sell 25% |
| Runner | Post-TP2 | Trail 25% with Chandelier (2.5× ATR) |

Backtest results (56-ticker universe, 2 years): **58 trades, 41% win rate, +0.5% avg return**.

---

### Setup C — Volatility Contraction *(in development)*
Tight coiling setups that precede explosive moves. Rare, high-conviction entries only.
- Bollinger Band width at 52-week percentile low
- ATR at 52-week percentile low
- 20-day range contraction
- Volume dry-up (below 50-day average)
- Trigger: breakout bar with volume > 2.5× average

Expected profile: lowest frequency, highest average winner.
Capital allocation: 20–30%.

---

### Momentum Screener *(live — benchmark strategy)*
A stripped 3-factor screener kept alongside the event-based engines as an experimental baseline.
- EMA9 > EMA21 (trend)
- Relative volume > 2× average
- Gap > 3%

Scored 0–100; candidates above threshold advance to Claude AI Stage-2 scoring.
Backtest results: **170 trades, 49% win rate, +0.1% avg return**.

---

### Crypto Mean Reversion *(live)*
RSI + Bollinger Band oversold bounce on 13 Alpaca-supported crypto assets.
Backtest results: **86 trades, 70% win rate, +2.9% avg return**.

---

## Backtest Results Summary

All results from a 2-year walk-forward backtest across a 170-stock + 8-crypto universe.

| Strategy | Trades | Win Rate | Avg Return | Profit Factor |
|----------|--------|----------|------------|---------------|
| Chart Pattern | 2,186 | 52% | +1.7% | 2.1 |
| Crypto Mean Reversion | 86 | 70% | +2.9% | 3.8 |
| Institutional Breakout | 58 | 41% | +0.5% | 1.6 |
| Momentum Screener | 170 | 49% | +0.1% | 1.1 |

**Combined Monte Carlo (1,000 paths, $100/trade):**
- Median return: **+721.9%**
- 5th percentile (worst realistic): -56.9%
- 95th percentile (best realistic): +17,977%
- Probability of ruin: **2.7%**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  Alpaca Market Data → Tiingo → yfinance (fallback chain)    │
│  FMP (earnings, float, RS universe) · FinnHub (sentiment)   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     ALPHA ENGINES                           │
│  Setup B (IB) ──┐                                           │
│  Momentum    ───┼──► Signal Router (IB wins on conflict)    │
│  Chart Pattern  │                                           │
│  Crypto MeanRev ┘                                           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   MARKET REGIME GATE                        │
│  SPY slope · ADX · QQQ leadership · VIX breadth            │
│  bear/strong_bear → all entries blocked                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│               CLAUDE AI STAGE-2 SCORING                     │
│  Shortlisted tickers → narrative quality gate               │
│  FinnHub sentiment · earnings surprise · signal quality     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     EXECUTION                               │
│  Alpaca (paper / live) · Bracket orders · ATR sizing        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    EXIT MANAGER                             │
│  Strategy-specific exit profiles · Chandelier trail        │
│  FinnHub news gate · 8-K detection · time stop             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
quantbot/
├── main.py                      # Entry point — scheduler, orchestration
├── scan_scheduler.py            # Scheduled scan runner
├── webhook_server.py            # TradingView + FinnHub webhook receiver (Flask)
├── telegram_bot.py              # Telegram command interface
├── dashboard_server.py          # Flask performance dashboard
├── config.py                    # Centralised config (reads .env)
│
├── agents/
│   ├── institutional_breakout.py  # Setup B screener (live)
│   ├── momentum_screener.py       # 3-factor momentum screen
│   ├── chart_pattern_screener.py  # Candlestick pattern detection
│   ├── crypto_mean_reversion.py   # RSI + BB oversold crypto entries
│   ├── signal_router.py           # Multi-engine deduplication + priority
│   ├── exit_manager.py            # Strategy-specific exit profiles
│   ├── market_regime.py           # SPY/QQQ/VIX regime classifier
│   ├── stock_screener.py          # Unified screener (Stage 1)
│   ├── tiingo_screener.py         # Tiingo chart-pattern screen
│   ├── vwap_screener.py           # VWAP pullback scanner
│   └── options_flow.py            # Unusual Whales options flow
│
├── data/
│   ├── alpaca_data.py             # Alpaca Market Data API (primary OHLCV)
│   ├── tiingo_data.py             # Tiingo REST API (secondary OHLCV)
│   ├── market_data.py             # Unified data layer with fallback chain
│   ├── fmp_data.py                # Financial Modeling Prep (RS universe, float)
│   ├── finnhub_data.py            # FinnHub (news sentiment, earnings surprise)
│   ├── exit_state.py              # Per-position strategy tagging
│   ├── custom_watchlist.py        # Your custom stock watchlist
│   ├── crypto_data.py             # Crypto watchlist + helpers
│   └── universe.py                # S&P 500 / Nasdaq / Russell universe fetchers
│
├── brokers/
│   ├── alpaca.py                  # Order placement, positions, account
│   └── alpaca_options.py          # Options order handling
│
├── utils/
│   ├── trade_journal.py           # SQLite trade log + performance_by_strategy()
│   ├── position_monitor.py        # Open position tracker + self-healing
│   ├── risk.py                    # Position size limits, daily loss guard
│   ├── preflight.py               # Startup dependency health checks
│   ├── autoresearch.py            # Weekly indicator hypothesis testing loop
│   ├── claude_client.py           # Anthropic API wrapper
│   ├── metrics.py                 # Prometheus-style counters
│   ├── version.py                 # Git hash version stamp on startup
│   └── logging.py                 # Structured JSON logging (loguru)
│
├── backtesting/
│   ├── run_all_backtests.py       # Full suite runner
│   ├── institutional_breakout_backtest.py
│   ├── chart_pattern_backtest.py
│   ├── crypto_mean_reversion.py
│   ├── screener_backtest.py
│   ├── monte_carlo.py             # 1,000-path Monte Carlo simulator
│   ├── walk_forward.py            # IS/OOS walk-forward validator
│   └── _tiingo_compat.py          # Alpaca → Tiingo → yfinance download shim
│
├── requirements.txt
├── .env.example                   # Copy to .env and fill in credentials
├── Dockerfile
└── docker-compose.yml
```

---

## Daily Scan Schedule (Mountain Time)

| Time | Scan |
|------|------|
| 08:00 | Full watchlist screen (unified scorer + Claude AI) |
| 08:05 | Dow 30 screen |
| 08:15 | Nasdaq top 30 screen |
| 08:25 | Small/mid-cap screen |
| 08:40 | VWAP pullback scan |
| 08:55 | Options flow scan (Unusual Whales) |
| 09:00 | Chart pattern scan |
| 09:05 | Tiingo scored screen |
| 09:10 | **Institutional Breakout scan (Setup B)** |
| 13:30 | Pre-close watchlist rescan |
| 13:35 | Pre-close chart patterns |
| 13:40 | Pre-close Tiingo screen |
| 13:45 | Pre-close IB scan |
| 14:05 | Cancel unfilled buy orders |
| Every 3h | Crypto mean reversion scan (24/7) |
| Mon 08:45 | Autoresearch: generate weekly hypothesis |
| Fri 15:45 | Autoresearch: evaluate + commit or revert |

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Alpaca account](https://alpaca.markets) — paper trading is free
- [Anthropic API key](https://console.anthropic.com) — for Claude AI Stage-2 scoring
- Telegram bot (optional but recommended) — for remote control and alerts

**Optional (enhance signal quality):**
- [Tiingo API key](https://tiingo.com) — secondary OHLCV source
- [Financial Modeling Prep API key](https://financialmodelingprep.com) — RS universe, float data
- [FinnHub API key](https://finnhub.io) — news sentiment, earnings surprises

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/quantbot.git
cd quantbot
```

### 2. Create a virtual environment

```bash
# Using uv (recommended — much faster)
pip install uv
uv venv .venv
source .venv/bin/activate        # macOS/Linux
.\.venv\Scripts\activate         # Windows

# Install dependencies
uv pip install -r requirements.txt
```

**Or using standard pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials. At minimum you need:
```env
ANTHROPIC_API_KEY=sk-ant-...      # Claude AI scoring
ALPACA_API_KEY=PK...               # Alpaca paper trading
ALPACA_SECRET_KEY=...
```

Everything else is optional — the bot runs without Tiingo, FMP, FinnHub, and Telegram, it just loses those signal enhancement layers.

### 4. Run the preflight check

```bash
python selftest.py
```

This verifies all configured API keys, tests data provider connectivity, and confirms the bot can reach Alpaca. Fix any `[FAIL]` items before starting the bot.

### 5. Start the bot

```bash
python main.py
```

Or run each service independently:
```bash
python main.py           # Core trading engine + scheduler
python webhook_server.py # TradingView / FinnHub webhook receiver (port 8080)
python telegram_bot.py   # Telegram command interface (port 9100)
python dashboard_server.py # Performance dashboard (port 5001)
```

### 6. Run the backtest suite

```bash
python backtesting/run_all_backtests.py
```

Add `--skip-walk-forward` for a faster run (skips the 12-month IS / 3-month OOS walk-forward validation):

```bash
python backtesting/run_all_backtests.py --skip-walk-forward
```

---

## Telegram Commands

Once your Telegram bot is configured, control the system from anywhere:

| Command | Action |
|---------|--------|
| `/scan` | Run full morning scan on demand |
| `/ib` | Run Institutional Breakout scan (Setup B) |
| `/tiingo` | Run Tiingo chart-pattern screen |
| `/positions` | Show open positions and P&L |
| `/account` | Portfolio summary |
| `/perf` | Strategy attribution (win rates by engine) |
| `/add TICKER` | Add ticker to custom watchlist |
| `/remove TICKER` | Remove ticker from custom watchlist |
| `/watchlist` | Show current custom watchlist |
| `/help` | List all commands |

---

## Webhook Integration

### TradingView Alerts

Point a TradingView alert to `https://your-ngrok-url/webhook` with JSON body:

```json
{
  "ticker": "NVDA",
  "action": "SCAN",
  "notional": 100,
  "secret": "your-webhook-secret"
}
```

**Actions:** `SCAN` (screen first, buy if passes), `BUY` (direct buy), `SELL`.

### FinnHub Real-Time Events

Configure a FinnHub webhook at `https://your-ngrok-url/finnhub` to receive:
- **News events** → fetches signal quality score, Telegrams you headline + gate status
- **Earnings events** → shows EPS actual vs estimate, surprise %, resets AVWAP anchor
- **Recommendation changes** → analyst rating updates for watchlist tickers

---

## Market Regime

The regime model gates all entries. In bear or strong-bear conditions, no new long positions are opened regardless of signal strength.

```
Inputs:   SPY 20-day slope · ADX(14) · QQQ 20-day slope · VIX level
                      SPY > 50 EMA
Output:   Regime:  strong_bull | bull | neutral | bear | strong_bear
          Breadth: 0.25× | 0.50× | 0.75× | 1.00× position size multiplier
```

The breadth multiplier stacks with the strategy-level position size — a stock scoring at 75% in a neutral regime (0.75× multiplier) gets 56% of base notional.

---

## Configuration Reference

All settings live in `.env`. Key parameters:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | `dev` (paper) or `prod` (live) | `dev` |
| `ANTHROPIC_API_KEY` | Claude AI API key | required |
| `ALPACA_API_KEY` | Alpaca API key | required |
| `ALPACA_SECRET_KEY` | Alpaca secret key | required |
| `ALPACA_BASE_URL` | Alpaca endpoint | paper |
| `TIINGO_API_KEY` | Tiingo (secondary OHLCV) | optional |
| `FMP_API_KEY` | Financial Modeling Prep | optional |
| `FINNHUB_API_KEY` | FinnHub news/sentiment | optional |
| `FINNHUB_WEBHOOK_SECRET` | FinnHub webhook auth | optional |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | optional |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | optional |
| `WEBHOOK_SECRET` | TradingView webhook auth | recommended |
| `MAX_POSITION_SIZE_USD` | Max per-trade notional | `500` |
| `DAILY_LOSS_LIMIT_USD` | Daily loss circuit breaker | `100` |

---

## Signal Flow

```
Market open
     │
     ▼
Stage 1: Screener (threshold gate)
     │  score < BUY threshold → skip
     ▼
Stage 2: Claude AI narrative scoring
     │  score < AI threshold → skip
     ▼
Signal Router: deduplication
     │  IB signal wins over momentum on same ticker
     ▼
Market Regime: bull/bear gate
     │  bear/strong_bear → block all longs
     ▼
Risk check: position size + daily loss limit
     │  limit breached → skip
     ▼
Alpaca: bracket order (entry + stop + optional TP)
     │
     ▼
Exit Manager: monitors every 30 minutes
     ├── IB positions:        2ATR / 4ATR partial → Chandelier trail
     ├── Momentum positions:  fixed % stops
     └── Crypto positions:    ATR-based exits
```

---

## Adding to Your Watchlist

Edit `data/custom_watchlist.py` directly, or use Telegram:

```
/add PLTR
/add CRWD
/add HIMS
```

The watchlist persists across restarts. Tickers are scanned by every engine on every scan cycle.

---

## Safety and Risk Management

- **Paper trading by default** — `ENV=dev` hard-enforces the paper endpoint even if you accidentally set the live URL
- **Preflight gate** — bot refuses to start if data providers fail health checks
- **Fail-closed exits** — if Alpaca API is unreachable, the exit manager logs FAIL-CLOSED and skips (no blind selling)
- **Daily loss limit** — once `DAILY_LOSS_LIMIT_USD` is hit, no new buys for the rest of the day
- **Max position size** — no single trade can exceed `MAX_POSITION_SIZE_USD`
- **Duplicate prevention** — bot checks existing positions before placing any new buy
- **Watchdog** — separate watchdog process restarts crashed services automatically
- **Global mutex** — prevents two bot instances from running simultaneously

---

## Running in Docker

```bash
docker-compose up -d
```

The docker-compose file starts all four services (bot, webhook, telegram, dashboard) with environment variables passed from `.env`.

---

## Overfitting Prevention

This bot is built with walk-forward validation as the standard evaluation method — not in-sample optimisation.

Key principles applied throughout:
- **Fewer factors outperform more factors** — the momentum screener uses 3 indicators (stripped from 11) based on walk-forward evidence showing the full model degrades OOS
- **Regime segmentation** — chart patterns were found to lose money (profit factor ~0.28) in bear regimes and hold up in bull; all strategies are regime-gated
- **OOS validation** — every threshold change is tested on held-out data before deployment
- **Attribution by design** — exit tagging, strategy columns in the journal, and `performance_by_strategy()` ensure alpha is traceable to its source

---

## Disclaimer

Im always open for ideas and suggestions.

This software is for **educational and research purposes only**. It is not financial advice. Paper trading results do not guarantee live trading performance. Trading involves substantial risk of loss. Use at your own risk.

The default configuration uses Alpaca **paper trading** which uses simulated money only. You must explicitly set `ENV=prod` and the live Alpaca endpoint to trade with real capital — do so only if you fully understand the risks.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
